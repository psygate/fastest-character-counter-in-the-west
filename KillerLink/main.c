/*

Author: Theodor Mittermair
Date: 2020-06-12
Compilation:
	gcc -Wall -Werror -pedantic -O3 main.c -o main
Running:
	./main 100000000 10
Notes:
	Many thanks to usefull material available under:
		https://godbolt.org
		https://cdecl.org
		https://software.intel.com/sites/landingpage/IntrinsicsGuide/
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <emmintrin.h>

static inline void solution1(char* restrict data, int length, int result[4]) {
	int bins[4] = {0,0,0,0};
	for (int i=0; i<length; i++) {
		switch (data[i]) {
			case 'A': bins[0]++; break;
			case 'C': bins[1]++; break;
			case 'G': bins[2]++; break;
			case 'T': bins[3]++; break;
		}
	}
	result[0] = bins[0];
	result[1] = bins[1];
	result[2] = bins[2];
	result[3] = bins[3];
}

static inline void solution2(const char* restrict data, const int length, int result[4]) {
	int bins[256];
	memset(bins,0,256*sizeof(int));
	for (int i=0; i<length; i++) {
		bins[(int)data[i]]++;
	}
	result[0] = bins['A'];
	result[1] = bins['C'];
	result[2] = bins['G'];
	result[3] = bins['T'];
}

static inline void solution3(const char* restrict data, const int length, int ret[4]) {
	for (int i=0; i<4; i++) {
		ret[i] = 0;
	}

	unsigned char isA[16], isC[16], isG[16], isT[16];
	memset(isA,'A',16*sizeof(char));
	memset(isC,'C',16*sizeof(char));
	memset(isG,'G',16*sizeof(char));
	memset(isT,'T',16*sizeof(char));

	__m128i vData, vA, vC, vG, vT;

	vA = _mm_loadu_si128((__m128i*)isA);
	vC = _mm_loadu_si128((__m128i*)isC);
	vG = _mm_loadu_si128((__m128i*)isG);
	vT = _mm_loadu_si128((__m128i*)isT);

	for (int i=0; i<length/16; i++) {
		vData = _mm_loadu_si128((__m128i*)&data[i*16]);
		ret[0]+=__builtin_popcount(_mm_movemask_epi8(_mm_cmpeq_epi8(vData,vA)));
		ret[1]+=__builtin_popcount(_mm_movemask_epi8(_mm_cmpeq_epi8(vData,vC)));
		ret[2]+=__builtin_popcount(_mm_movemask_epi8(_mm_cmpeq_epi8(vData,vG)));
		ret[3]+=__builtin_popcount(_mm_movemask_epi8(_mm_cmpeq_epi8(vData,vT)));
	}

}

static inline void solution4(const char* restrict data, const int length, int result[4]) {

	unsigned char isA[16], isC[16], isG[16], isT[16], is0[16];
	memset(is0,0,16*sizeof(char));
	memset(isA,'A',16*sizeof(char));
	memset(isC,'C',16*sizeof(char));
	memset(isG,'G',16*sizeof(char));
	memset(isT,'T',16*sizeof(char));

	__m128i vData;
	__m128i v0, vA, vC, vG, vT, vR;
	__m128i vsA, vsC, vsG, vsT;

	v0 = _mm_loadu_si128((__m128i*)is0);
	vA = _mm_loadu_si128((__m128i*)isA);
	vC = _mm_loadu_si128((__m128i*)isC);
	vG = _mm_loadu_si128((__m128i*)isG);
	vT = _mm_loadu_si128((__m128i*)isT);

	union {
		uint8_t u8[16];
		uint16_t u16[8];
	} r;

	int cA=0; int cC=0; int cG=0; int cT=0;

	for (int i=0; i<length/16/128; i++) {
		vsA = v0;
		vsC = v0;
		vsG = v0;
		vsT = v0;
		for (int j=0; j<128; j++) {
			int idx = i*16*128+j*16;
			vData = _mm_loadu_si128((__m128i*)&data[idx]);
			vsA = _mm_sub_epi8(vsA,_mm_cmpeq_epi8(vData,vA));
			vsC = _mm_sub_epi8(vsC,_mm_cmpeq_epi8(vData,vC));
			vsG = _mm_sub_epi8(vsG,_mm_cmpeq_epi8(vData,vG));
			vsT = _mm_sub_epi8(vsT,_mm_cmpeq_epi8(vData,vT));
		}
		vR = _mm_sad_epu8(v0,vsA);
		_mm_storeu_si128((__m128i*)&r.u8[0],vR);
		cA+=r.u16[0]+r.u16[4];
		vR = _mm_sad_epu8(v0,vsC);
		_mm_storeu_si128((__m128i*)&r.u8[0],vR);
		cC+=r.u16[0]+r.u16[4];
		vR = _mm_sad_epu8(v0,vsG);
		_mm_storeu_si128((__m128i*)&r.u8[0],vR);
		cG+=r.u16[0]+r.u16[4];
		vR = _mm_sad_epu8(v0,vsT);
		_mm_storeu_si128((__m128i*)&r.u8[0],vR);
		cT+=r.u16[0]+r.u16[4];
	}


	memset(result,0,4*sizeof(int));
	solution2(&data[length-length%2048],length%2048,result);
	result[0]+=cA;
	result[1]+=cC;
	result[2]+=cG;
	result[3]+=cT;

}

static inline void solution5(const char* restrict data, const int length, int result[4]) {
	#define UFC_S5 4
	const char chars[4] = "ACGT";
	int bins[256][UFC_S5];
	memset(bins,0,UFC_S5*256*sizeof(int));
	for (int i=0; i<length; i++) {
		bins[(int)data[i]][i%UFC_S5]++;
	}
	for (int i=0; i<4; i++) {
		result[i] = 0;
		for (int j=0; j<UFC_S5; j++) {
			result[i] += bins[(int)chars[i]][j];
		}
	}
}

static inline void solution6(const char* restrict data, const int length, int result[4]) {
	#define UFC_S6 4
	const char chars[4] = "ACGT";
	int bins[UFC_S6][256];
	memset(bins,0,UFC_S6*256*sizeof(int));
	for (int i=0; i<length; i++) {
		bins[i%UFC_S6][(int)data[i]]++;
	}
	for (int i=0; i<4; i++) {
		result[i] = 0;
		for (int j=0; j<UFC_S6; j++) {
			result[i] += bins[j][(int)chars[i]];
		}
	}
}

static inline void solution7(const char* restrict data, const int length, int result[4]) {
	int bins[4];
	memset(bins,0,4*sizeof(int));
	for (int i=0; i<length; i+=2) {
		bins[0]+= data[i]=='A';
		bins[1]+= data[i]=='C';
		bins[2]+= data[i]=='G';
		bins[3]+= data[i]=='T';
	}
	result[0] = bins[0];
	result[1] = bins[1];
	result[2] = bins[2];
	result[3] = bins[3];
}

static inline void solution8(const char* restrict data, const int length, int result[4]) {
	#define UFC_S8 4
	#define J_FIRST
	int bins[UFC_S8][4];
	memset(bins,0,UFC_S8*4*sizeof(int));
	for (int i=0; i<length; i+=UFC_S8) {
		for (int j=0; j<UFC_S8; j++) {
			#ifdef J_FIRST
			bins[j][0] += data[i+j]=='A';
			bins[j][1] += data[i+j]=='C';
			bins[j][2] += data[i+j]=='G';
			bins[j][3] += data[i+j]=='T';
			#else
			bins[0][j] += data[i+j]=='A';
			bins[1][j] += data[i+j]=='C';
			bins[2][j] += data[i+j]=='G';
			bins[3][j] += data[i+j]=='T';
			#endif
		}
	}
	for (int i=0; i<4; i++) {
		result[i] = 0;
		for (int j=0; j<UFC_S8; j++) {
			#ifdef J_FIRST
			result[i] += bins[j][i];
			#else
			result[i] += bins[i][j];
			#endif
		}
	}
}


int main(int argc, char** argv) {

	if (argc != 3) {
		printf("usage: %s <length> <samples>\n", argv[0]);
		return EXIT_FAILURE;
	}
	char* end;

	int num_length = strtol(argv[1], &end, 10);
	if (*end!='\0') {
		printf("error converting length argument\n");
		return EXIT_FAILURE;
	}

	int num_samples = strtol(argv[2], &end, 10);
	if (*end!='\0') {
		printf("error converting length argument\n");
		return EXIT_FAILURE;
	}

	srandom(1234567890);

	char* data_x = malloc(num_length*sizeof(char)+8);
	char* data = (char*)(((long)data_x+8)&(~0x7));

	for (int i=0; i<num_length; i++) {
		char chars[4] = "ACGT";
		data[i] = chars[random()%4];
	}


	//num_length = 70;
	//data = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC";
	//#include "testcase.h"
	//num_length = testlength-70;
	//data = testdata;

	int result[4] = {0,0,0,0};

	struct timespec start, stop;
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
		perror("clock gettime");
		return EXIT_FAILURE;
	}
	for (int i=0; i<num_samples; i++) {
		solution4(data,num_length,result);
	}
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
		perror("clock gettime");
		return EXIT_FAILURE;
	}
	printf("ACGT = %d,%d,%d,%d\n",result[0],result[1],result[2],result[3]);
	double time = (stop.tv_sec - start.tv_sec)
		+(double)(stop.tv_nsec - start.tv_nsec)/1000000000.0;

	printf("Total: %ld\n", (long)(result[0]+result[1]+result[2]+result[3]));

   	printf( "Average Time: %lf seconds\n", time/num_samples);

	free(data_x);

	return EXIT_SUCCESS;
}
