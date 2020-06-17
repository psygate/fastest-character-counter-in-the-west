#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <emmintrin.h>    // SSE2
#include <xmmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <malloc.h>

typedef int64_t ctr_t;

union int64_256
{
    __m256i value;
    uint64_t parts[4];
};

struct result {
    int a;
    int t;
    int c;
    int g;
};


int generate(char* array);
void fill(char* data,const size_t size);
void count(const char* data, size_t size, ctr_t* output);
void count1(const char* data, size_t size, ctr_t* output);
void countAVX2(const char* data, size_t size, ctr_t* output) ;
void countAVX2_1(const char* data, size_t size, ctr_t* output) ;
static void count_nspace(const char *data, size_t size, struct result* out);

void dump_value_m256i(const char *str, __m256i value);
void dump_value_m128i(const char *str, __m128i value);
void dump_value_m128(const char *str, __m128 value);

static const char *ALPHABET = "ACGT";

int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
           ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}

static __m256i REFS[sizeof(ALPHABET)];
static __m256i A_REF, C_REF, G_REF, T_REF;


int main(int argc, char** argv) {
    for(int i = 0; i < 4; i++) {
        memset(REFS + i, ALPHABET[i], sizeof(__m256i));
    }

    A_REF = REFS[0];
    C_REF = REFS[1];
    G_REF = REFS[2];
    T_REF = REFS[3];

    size_t psize = strtol(argv[1], NULL, 10);
    printf("Using size: %lu\n", psize);


    for(int i = 0; i < 16; i++) 
    {
        void* original = malloc(psize + 255);
        void* o = (void *)(((uintptr_t)(original + 255)) & ~(uintptr_t)255);
        char* data = (char*)o;

        fill(data, psize);

        {       
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            ctr_t values[255];
            memset(values, 0, sizeof(values));
            count(data, psize, values);
            clock_gettime(CLOCK_MONOTONIC, &end);

            uint64_t timeElapsed = timespecDiff(&end, &start);
            printf("count:        %lums A: %lld, C: %lld, G: %lld, T: %lld\n", 
                timeElapsed / 1000000,
                values['A'],
                values['C'],
                values['G'],
                values['T']
            );
        }
        {       
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            ctr_t values[255];
            memset(values, 0, sizeof(values));
            countAVX2(data, psize, values);
            clock_gettime(CLOCK_MONOTONIC, &end);

            uint64_t timeElapsed = timespecDiff(&end, &start);
            printf("countAVX2:    %lums A: %lld, C: %lld, G: %lld, T: %lld\n", 
                timeElapsed / 1000000,
                values['A'],
                values['C'],
                values['G'],
                values['T']
            );       
        }
        {       
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            ctr_t values[255];
            memset(values, 0, sizeof(values));
            countAVX2_1(data, psize, values);
            clock_gettime(CLOCK_MONOTONIC, &end);

            uint64_t timeElapsed = timespecDiff(&end, &start);
            printf("countAVX2_1:  %lums A: %lld, C: %lld, G: %lld, T: %lld\n", 
                timeElapsed / 1000000,
                values['A'],
                values['C'],
                values['G'],
                values['T']
            );       
        }
        {       
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            ctr_t values[255];
            memset(values, 0, sizeof(values));
            struct result out = {0};
            count_nspace(data, psize, &out);
            clock_gettime(CLOCK_MONOTONIC, &end);

            uint64_t timeElapsed = timespecDiff(&end, &start);
            printf("count_nspace: %lums A: %d, C: %d, G: %d, T: %d\n", 
                timeElapsed / 1000000,
                out.a,
                out.c,
                out.g,
                out.t
            );       
        }


        free(original);
        // free(original);
    }

    
    return EXIT_SUCCESS;
}

#define dump_value(str, value) {\
    printf("%s: ", str);\
    const unsigned char *ptr = (const unsigned char*) &value;\
    for(size_t i = 0; i < sizeof(value); i++)\
    {\
        printf("%02X ", ptr[i]);\
    }\
    printf("\n");\
}

void count(const char* data, size_t size, ctr_t* output) 
{
    for(size_t i = 0; i < size; i++) {
        output[(int)(data[i])] += 1;
    }
}

void dump_value_m256i(const char *str, __m256i value)
{
    printf("%s: ", str);
    char *ptr = (char*) &value;
    for(size_t i = 0; i < sizeof(__m256i); i++)
    {
        printf("%02X ", ptr[i]);
    }
    printf("\n");
}


void dump_value_m128i(const char *str, __m128i value)
{
    printf("%s: ", str);
    char *ptr = (char*) &value;
    for(size_t i = 0; i < sizeof(__m128i); i++)
    {
        printf("%02X ", (unsigned char) ptr[i]);
    }
    printf("\n");
}

void dump_value_m128(const char *str, __m128 value)
{
    printf("%s: ", str);
    char *ptr = (char*) &value;
    for(size_t i = 0; i < sizeof(__m128); i++)
    {
        printf("%02X ", (unsigned char) ptr[i]);
    }
    printf("\n");
}

static ctr_t hsum_(const __m256i value) 
{   
    ctr_t sum = 0;
    const uint8_t *ptr = (const uint8_t*) &value;

    for(size_t i = 0; i < sizeof(__m256i); i++)
    {
        sum += ptr[i];
    }

    return sum;
}

static inline int64_t hsum(const __m256i value) 
{   
    const __m256i comb = _mm256_sad_epu8(value, _mm256_setzero_si256());

    const __m128i hiQuad = _mm256_extracti128_si256(comb, 1);
    const __m128i loQuad = _mm256_castsi256_si128(comb);
    const __m128i sum = _mm_add_epi64(hiQuad, loQuad);
    const int64_t qhi = _mm_extract_epi64(sum, 1);
    const int64_t qlo = _mm_extract_epi64(sum, 0);
    const int64_t fsum = qhi + qlo;
    return fsum;
}

void countAVX2_1(const char* data, size_t size, ctr_t* output) 
{
    const size_t elements = size / sizeof(__m256i);
    const __m256i* mdata = (__m256i *) data;
    
    __m256i abuffer = _mm256_setzero_si256();
    __m256i cbuffer = _mm256_setzero_si256();
    __m256i gbuffer = _mm256_setzero_si256();
    __m256i tbuffer = _mm256_setzero_si256();

    size_t i;
    for(i = 0; i < elements; i++)
    {
        {
            const __m256i introut = _mm256_cmpeq_epi8( 
                A_REF,
                _mm256_loadu_si256(mdata + i)
            );

            abuffer = _mm256_subs_epi8(abuffer, introut);
       }
        {
            const __m256i introut = _mm256_cmpeq_epi8( 
                C_REF,
                _mm256_loadu_si256(mdata + i)
            );

            cbuffer = _mm256_subs_epi8(cbuffer, introut);
       }
        {
            const __m256i introut = _mm256_cmpeq_epi8( 
                G_REF,
                _mm256_loadu_si256(mdata + i)
            );

            gbuffer = _mm256_subs_epi8(gbuffer, introut);
       }

       if(i % 255 == 0)
       {           
            output['A'] += hsum(abuffer);
            output['C'] += hsum(cbuffer);
            output['G'] += hsum(gbuffer);

            abuffer = _mm256_setzero_si256();
            cbuffer = _mm256_setzero_si256();
            gbuffer = _mm256_setzero_si256();
            tbuffer = _mm256_setzero_si256();
       }
    }

    output['A'] += hsum(abuffer);
    output['C'] += hsum(cbuffer);
    output['G'] += hsum(gbuffer);

    for(size_t j = i * sizeof(__m256i); j < size; j++) {
        switch(data[j]) 
        {
            case 'A': output['A']++; break;
            case 'C': output['C']++; break;
            case 'G': output['G']++; break;
            case 'T': output['T']++; break;
        }
    }

    output['T'] += (size - (output['A'] + output['C'] + output['G']));
}

void countAVX2(const char* data, size_t size, ctr_t* output) 
{
    const size_t elements = size / sizeof(__m256i);
    const __m256i* mdata = (__m256i *) data;

    for(size_t i = 0; i < elements; i++)
    {
        {
            union int64_256 introut;
            introut.value = _mm256_cmpeq_epi8( 
                A_REF,
                _mm256_loadu_si256(mdata + i)
            ); 

            output['A'] += _mm_popcnt_u64(introut.parts[0]) / 8L;
            output['A'] += _mm_popcnt_u64(introut.parts[1]) / 8L;
            output['A'] += _mm_popcnt_u64(introut.parts[2]) / 8L;
            output['A'] += _mm_popcnt_u64(introut.parts[3]) / 8L;
        }
        {
            union int64_256 introut;
            introut.value = _mm256_cmpeq_epi8( 
                C_REF,
                _mm256_loadu_si256(mdata + i)
            ); 

            output['C'] += _mm_popcnt_u64(introut.parts[0]) / 8L;
            output['C'] += _mm_popcnt_u64(introut.parts[1]) / 8L;
            output['C'] += _mm_popcnt_u64(introut.parts[2]) / 8L;
            output['C'] += _mm_popcnt_u64(introut.parts[3]) / 8L;
        }
        {
            union int64_256 introut;
            introut.value = _mm256_cmpeq_epi8( 
                G_REF,
                _mm256_loadu_si256(mdata + i)
            ); 

            output['G'] += _mm_popcnt_u64(introut.parts[0]) / 8L;
            output['G'] += _mm_popcnt_u64(introut.parts[1]) / 8L;
            output['G'] += _mm_popcnt_u64(introut.parts[2]) / 8L;
            output['G'] += _mm_popcnt_u64(introut.parts[3]) / 8L;
        }
                {
            union int64_256 introut;
            introut.value = _mm256_cmpeq_epi8( 
                T_REF,
                _mm256_loadu_si256(mdata + i)
            ); 

            output['T'] += _mm_popcnt_u64(introut.parts[0]) / 8L;
            output['T'] += _mm_popcnt_u64(introut.parts[1]) / 8L;
            output['T'] += _mm_popcnt_u64(introut.parts[2]) / 8L;
            output['T'] += _mm_popcnt_u64(introut.parts[3]) / 8L;
        }
    }

    for(size_t i = elements * sizeof(__m256i); i < size; i++) {
        switch(data[i]) 
        {
            case 'A': output['A']++; break;
            case 'C': output['C']++; break;
            case 'G': output['G']++; break;
            case 'T': output['T']++; break;
        }
    }
}

/**
 * By @nspace#4141, the fast boi of the wolfstack docks.
 */
static void count_nspace(const char *data, size_t size, struct result* out)
{

    const __m256i *vector = (const __m256i *)data;
    for (int i = 0; i < (size / 32); i++) {
        __m256i v = _mm256_load_si256(vector);
        uint32_t c = _mm256_movemask_epi8(_mm256_slli_epi16(v, 5));
        uint32_t d = _mm256_movemask_epi8(_mm256_slli_epi16(v, 6));

        out->a += _mm_popcnt_u32(~(c | d));
        out->g += _mm_popcnt_u32(c & d);
        out->c += _mm_popcnt_u32((c ^ d) & d);
        out->t += _mm_popcnt_u32((c ^ d) & c);

        vector++;
    }
}

// void countAVX2(const char* data, size_t size, ctr_t* output) 
// {
//     const size_t elements = size / sizeof(__m256i);

//     for(size_t i = 0; i < elements; i++) {
//     {
//         for(int c = 0; c < sizeof(ALPHABET); c++)
//         {
//                 const __m256i introut = _mm256_cmpeq_epi8( 
//                     REFS[c],
//                     _mm256_loadu_si256((__m256i *)(data + (i * sizeof(__m256i))))
//                 ); 

//                 const uint64_t* intrptr64 = (uint64_t*)&introut;

//                 for(int j = 0; j < sizeof(__m256i) / sizeof(uint64_t); j++)
//                 {
//                     output[(int)ALPHABET[c]] += _mm_popcnt_u64(intrptr64[j]);
//                 }
//             }
//         }
//     }
// }

void fill(char* data,const size_t size) 
{
    srand(0x13377331);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = ALPHABET[rand() % strlen(ALPHABET)];
        // data[i] = 'A';
    }

}
