/**
 * @file    fir_baseline.c
 * @author  Vaidehi Gohil, Anthony Yalong
 * @brief   serial fir low-pass filter baseline (hamming window)
 * @usage   ./fir_baseline <input.bin> <n_samples> <num_taps> <cutoff_ratio> [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

// --- prototypes --------------------------------------------------------------

/**
 * @brief  compute elapsed time between two timespec values
 * @param  a  start time
 * @param  b  end time
 * @return elapsed time in milliseconds
 */
static double elapsed_ms(struct timespec a, struct timespec b);

/**
 * @brief  generate fir low-pass filter coefficients using the hamming window method;
 *         caller is responsible for freeing the returned array
 * @param  num_taps     number of filter taps, must be odd for linear phase
 * @param  cutoff_ratio normalized cutoff frequency in (0, 0.5), i.e. fc / fs
 * @return heap-allocated array of num_taps float coefficients, or NULL on failure
 */
static float *hamming_lowpass(int num_taps, double cutoff_ratio);

/**
 * @brief  apply fir filter h to input signal in, writing n output samples to out;
 *         samples before the start of in are treated as zero (zero-pad boundary)
 * @param  in       input signal, length n
 * @param  out      output signal, length n
 * @param  n        number of samples
 * @param  h        filter coefficient array, length num_taps
 * @param  num_taps number of filter taps
 */
static void fir_filter(const float *in, float *out, int n,
                       const float *h, int num_taps);

/**
 * @brief  print filtered output samples to stdout, one per line
 * @param  out  output sample array, length n
 * @param  n    number of samples
 */
static void print_output(const float *out, int n);

// --- helpers -----------------------------------------------------------------

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

// --- filter design -----------------------------------------------------------

// generate low-pass fir coefficients via hamming window method
// cutoff_ratio: normalized cutoff in (0, 0.5), i.e. fc / fs
static float *hamming_lowpass(int num_taps, double cutoff_ratio) {
    float *h = malloc(num_taps * sizeof(float));
    if (!h) { return NULL; }

    int M = num_taps - 1;   // filter order
    double sum = 0.0;

    for (int i = 0; i <= M; i++) {
        // hamming window weight at tap i
        double hamming = 0.54 - 0.46 * cos(2.0 * PI * i / M);

        // ideal low-pass sinc kernel, centered at M/2
        // center tap evaluated analytically to avoid divide-by-zero
        double sinc;
        if (i == M / 2) {
            sinc = 2.0 * cutoff_ratio;
        } else {
            sinc = sin(2.0 * PI * cutoff_ratio * (i - M / 2.0)) / (PI * (i - M / 2.0));
        }

        h[i] = (float)(hamming * sinc);
        sum += h[i];
    }

    // normalize so dc gain is unity
    for (int i = 0; i < num_taps; i++) {
        h[i] /= (float)sum;
    }

    return h;
}

// --- fir convolution ---------------------------------------------------------

// direct-form fir convolution; each output sample is independent of others
static void fir_filter(const float *in, float *out, int n,
                       const float *h, int num_taps) {
    for (int i = 0; i < n; i++) {
        double acc = 0.0;

        // dot product of filter coefficients with input window ending at i
        for (int k = 0; k < num_taps; k++) {
            int idx = i - k;
            // samples before the start of the array are treated as zero (zero-padding)
            if (idx >= 0) {
                acc += h[k] * in[idx];
            }
        }

        out[i] = (float)acc;
    }
}

// --- output ------------------------------------------------------------------

static void print_output(const float *out, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.6f\n", out[i]);
    }
}

// --- main --------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr,
            "usage: %s <input.bin> <n_samples> <num_taps> <cutoff_ratio> [-v]\n",
            argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int n            = atoi(argv[2]);
    int num_taps     = atoi(argv[3]);
    double cutoff    = atof(argv[4]);
    int verbose      = (argc == 6 && strcmp(argv[5], "-v") == 0);

    if (n <= 0 || num_taps <= 0 || cutoff <= 0.0 || cutoff >= 0.5) {
        fprintf(stderr, "error: n>0, num_taps>0, cutoff_ratio in (0, 0.5)\n");
        return 1;
    }

    // odd num_taps gives a symmetric (linear phase) filter
    if (num_taps % 2 == 0) {
        fprintf(stderr, "error: num_taps should be odd for symmetric filter\n");
        return 1;
    }

    // read float32 samples from binary file
    float *in = malloc(n * sizeof(float));
    if (!in) { perror("malloc"); return 1; }

    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); free(in); return 1; }
    if ((int)fread(in, sizeof(float), n, f) != n) {
        fprintf(stderr, "error: expected %d float32 samples\n", n);
        fclose(f); free(in); return 1;
    }
    fclose(f);

    float *out = malloc(n * sizeof(float));
    if (!out) { perror("malloc"); free(in); return 1; }

    // design filter outside timed region; only convolution is benchmarked
    float *h = hamming_lowpass(num_taps, cutoff);
    if (!h) { perror("malloc"); free(in); free(out); return 1; }

    // --- timed region --------------------------------------------------------
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    fir_filter(in, out, n, h, num_taps);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    // -------------------------------------------------------------------------

    printf("fir_baseline n=%d taps=%d time_ms=%.4f\n", n, num_taps, elapsed_ms(t0, t1));

    // dump output samples to stdout when -v is passed (for testing only)
    if (verbose) {
        print_output(out, n);
    }

    free(in); free(out); free(h);
    return 0;
}