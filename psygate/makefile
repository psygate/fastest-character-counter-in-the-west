CC=gcc
LIBS=
CFLAGS=-march=native -fopenmp -ffast-math -O3 -Wall

all: main.exe

main.exe: main.o
	$(CC) $(LIBS) $(CFLAGS) -S main.c 
	$(CC) $(LIBS) $(CFLAGS) -o main.exe main.o

main.o: main.c
	$(CC) $(LIBS) $(CFLAGS) -c main.c
     
clean:
	rm -f main.o main.exe main.s

.PHONY: clean all