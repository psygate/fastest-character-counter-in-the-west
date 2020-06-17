all: main

main:
	gcc -O3 -Wall -Werror -pedantic -o main main.c

run: main
	./main 100000000 20

main_profiled:
	rm -f main.gcda
	gcc -O3 -Wall -Werror -pedantic -fprofile-generate -o main_profiling main.c
	./main_profiling 204800000 10
	gcc -O3 -Wall -Werror -pedantic -fprofile-use -o main_profiled main.c
	./main_profiled 100000000 10

clean:
	rm -f ./main ./main_profiling ./main_profiled ./main.gcda
