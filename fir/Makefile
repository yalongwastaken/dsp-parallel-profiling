CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS = -lm

.PHONY: all clean

all: baseline/fir_baseline

baseline/fir_baseline: baseline/fir_baseline.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f baseline/fir_baseline