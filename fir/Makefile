CC      = gcc-15
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm

.PHONY: all clean

all: baseline/fir_baseline pthreads/fir_pthreads openmp/fir_openmp

baseline/fir_baseline: baseline/fir_baseline.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

pthreads/fir_pthreads: pthreads/fir_pthreads.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

openmp/fir_openmp: openmp/fir_openmp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

clean:
	rm -f baseline/fir_baseline pthreads/fir_pthreads openmp/fir_openmp