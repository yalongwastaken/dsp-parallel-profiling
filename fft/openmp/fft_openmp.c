/**
 * @file    fft_openmp.c
 * @author  Vaidehi Gohil, Anthony Yalong
 * @brief   openmp parallel cooley-tukey fft
 * @usage   ./fft_openmp <input.bin> <n_samples> <num_threads> [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define PI 3.14159265358979323846

typedef struct {
    double re;
    double im;
} complex_t;

// --- prototypes --------------------------------------------------------------

/**
 * @brief  compute elapsed time between two timespec values
 * @param  a  start time
 * @param  b  end time
 * @return elapsed time in milliseconds
 */
static double elapsed_ms(struct timespec a, struct timespec b);

/**
 * @brief  reorder elements of x into bit-reversed index order in-place;
 *         called single-threaded before parallel butterfly stages begin
 * @param  x  array of complex samples, length n
 * @param  n  number of samples, must be a power of 2
 */
static void bit_reverse(complex_t *x, int n);

/**
 * @brief  compute the in-place parallel cooley-tukey fft of x using openmp;
 *         each butterfly stage parallelizes the independent group iterations
 *         with an implicit barrier between stages via #pragma omp for
 * @param  x           complex sample array, length n; overwritten in-place
 * @param  n           number of samples, must be a power of 2
 * @param  num_threads number of openmp threads to use
 */
static void fft_parallel(complex_t *x, int n, int num_threads);

/**
 * @brief  print fft output magnitudes to stdout, one per line
 * @param  x  frequency-domain complex array, length n
 * @param  n  number of samples
 */
static void print_output(const complex_t *x, int n);

// --- helpers -----------------------------------------------------------------

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

// --- fft ---------------------------------------------------------------------

static void bit_reverse(complex_t *x, int n) {
    // count how many bits are needed to index n elements
    int bits = 0;
    while ((1 << bits) < n) {
        bits++;
    }

    for (int i = 0; i < n; i++) {
        // compute bit-reversal of i
        int j = 0;
        for (int b = 0; b < bits; b++) {
            if (i & (1 << b)) {
                j |= (1 << (bits - 1 - b));
            }
        }

        // swap each pair once to avoid swapping back
        if (j > i) {
            complex_t tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
}

static void fft_parallel(complex_t *x, int n, int num_threads) {
    bit_reverse(x, n);

    omp_set_num_threads(num_threads);

    // spawn thread team once and reuse across all stages via #pragma omp for;
    // the implicit barrier at the end of each omp for ensures all threads finish
    // a stage before any thread begins the next
    #pragma omp parallel
    {
        for (int len = 2; len <= n; len <<= 1) {
            double    ang  = -2.0 * PI / len;
            complex_t wlen = { cos(ang), sin(ang) };

            // parallelize across independent groups within this stage
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i += len) {
                complex_t w = { 1.0, 0.0 };     // twiddle factor, starts at w^0

                // butterfly between upper and lower halves of this group
                for (int j = 0; j < len / 2; j++) {
                    complex_t u = x[i + j];
                    // multiply lower-half element by current twiddle factor
                    complex_t v = {
                        x[i + j + len/2].re * w.re - x[i + j + len/2].im * w.im,
                        x[i + j + len/2].re * w.im + x[i + j + len/2].im * w.re
                    };

                    // combine into two output elements (butterfly operation)
                    x[i + j]         = (complex_t){ u.re + v.re, u.im + v.im };
                    x[i + j + len/2] = (complex_t){ u.re - v.re, u.im - v.im };

                    // rotate twiddle factor by wlen for next butterfly pair
                    double tmp = w.re * wlen.re - w.im * wlen.im;
                    w.im       = w.re * wlen.im + w.im * wlen.re;
                    w.re       = tmp;
                }
            }
            // implicit barrier here — all threads finish this stage before next
        }
    }
}

// --- output ------------------------------------------------------------------

static void print_output(const complex_t *x, int n) {
    for (int i = 0; i < n; i++) {
        double mag = sqrt(x[i].re * x[i].re + x[i].im * x[i].im);
        printf("%.6f\n", mag);
    }
}

// --- main --------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <input.bin> <n_samples> <num_threads> [-v]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int n            = atoi(argv[2]);
    int num_threads  = atoi(argv[3]);
    int verbose      = (argc == 5 && strcmp(argv[4], "-v") == 0);

    // cooley-tukey requires power-of-2 input length
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "error: n_samples must be a power of 2\n");
        return 1;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "error: num_threads must be > 0\n");
        return 1;
    }

    // read float32 samples from binary file
    float *buf = malloc(n * sizeof(float));
    if (!buf) { perror("malloc"); return 1; }

    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); free(buf); return 1; }
    if ((int)fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "error: expected %d float32 samples\n", n);
        fclose(f); free(buf); return 1;
    }
    fclose(f);

    // promote float32 input to double complex; imaginary parts are zero (real signal)
    complex_t *x = malloc(n * sizeof(complex_t));
    if (!x) { perror("malloc"); free(buf); return 1; }
    for (int i = 0; i < n; i++) {
        x[i] = (complex_t){ buf[i], 0.0 };
    }
    free(buf);

    // --- timed region --------------------------------------------------------
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    fft_parallel(x, n, num_threads);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    // -------------------------------------------------------------------------

    printf("fft_openmp n=%d threads=%d time_ms=%.4f\n", n, num_threads, elapsed_ms(t0, t1));

    // dump magnitudes to stdout when -v is passed (for testing only)
    if (verbose) {
        print_output(x, n);
    }

    free(x);
    return 0;
}