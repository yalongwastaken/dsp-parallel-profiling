CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm

.PHONY: all clean

all: baseline/fir_baseline pthreads/fir_pthreads

baseline/fir_baseline: baseline/fir_baseline.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

pthreads/fir_pthreads: pthreads/fir_pthreads.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

clean:
	rm -f baseline/fir_baseline pthreads/fir_pthreads