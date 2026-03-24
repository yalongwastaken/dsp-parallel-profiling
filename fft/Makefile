CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm

.PHONY: all clean

all: baseline/fft_baseline pthreads/fft_pthreads

baseline/fft_baseline: baseline/fft_baseline.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

pthreads/fft_pthreads: pthreads/fft_pthreads.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f baseline/fft_baseline pthreads/fft_pthreads