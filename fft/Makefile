CC      = gcc-15
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm

.PHONY: all clean

all: baseline/fft_baseline pthreads/fft_pthreads openmp/fft_openmp

baseline/fft_baseline: baseline/fft_baseline.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

pthreads/fft_pthreads: pthreads/fft_pthreads.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

openmp/fft_openmp: openmp/fft_openmp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

clean:
	rm -f baseline/fft_baseline pthreads/fft_pthreads openmp/fft_openmp