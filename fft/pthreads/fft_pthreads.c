/**
 * @file    fft_pthreads.c
 * @author  Vaidehi Gohil, Anthony Yalong
 * @brief   pthreads parallel cooley-tukey fft
 * @usage   ./fft_pthreads <input.bin> <n_samples> <num_threads> [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define PI 3.14159265358979323846

typedef struct { double re, im; } complex_t;

// --- shared state ------------------------------------------------------------

// precomputed parameters for each butterfly stage; avoids main/worker races
typedef struct {
    int    len;     // butterfly width for this stage
    double ang;     // twiddle base angle: -2*pi / len
} stage_t;

// all threads share this; set once before threads are spawned
typedef struct {
    complex_t *x;           // signal array (shared, modified in-place)
    int        n;           // total number of samples
    int        num_threads; // total thread count
    int        num_stages;  // total butterfly stages = log2(n)
    stage_t   *stages;      // precomputed stage parameters
} fft_shared_t;

// reusable barrier for synchronizing threads between butterfly stages
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    int             count;      // threads that have arrived
    int             total;      // total threads expected
    int             generation; // flips each release to guard spurious wakeups
} barrier_t;

// per-thread argument passed at spawn time
typedef struct {
    int           tid;
    fft_shared_t *shared;
    barrier_t    *barrier;
} thread_arg_t;

// --- prototypes --------------------------------------------------------------

/**
 * @brief  compute elapsed time between two timespec values
 * @param  a  start time
 * @param  b  end time
 * @return elapsed time in milliseconds
 */
static double elapsed_ms(struct timespec a, struct timespec b);

/**
 * @brief  initialize a reusable barrier for n threads
 * @param  b      barrier to initialize
 * @param  total  number of threads that must call barrier_wait before release
 */
static void barrier_init(barrier_t *b, int total);

/**
 * @brief  destroy mutex and condition variable held by barrier
 * @param  b  barrier to destroy
 */
static void barrier_destroy(barrier_t *b);

/**
 * @brief  block until all threads have arrived for this generation
 * @param  b  shared barrier
 */
static void barrier_wait(barrier_t *b);

/**
 * @brief  reorder elements of x into bit-reversed index order in-place;
 *         called single-threaded before parallel butterfly stages begin
 * @param  x  array of complex samples, length n
 * @param  n  number of samples, must be a power of 2
 */
static void bit_reverse(complex_t *x, int n);

/**
 * @brief  thread worker: processes a strided slice of butterfly groups
 *         for every stage, waiting at the barrier between each stage
 * @param  arg  pointer to thread_arg_t
 * @return NULL
 */
static void *fft_thread(void *arg);

/**
 * @brief  spawn num_threads workers to compute the in-place parallel fft of x
 * @param  x           complex sample array, length n; overwritten in-place
 * @param  n           number of samples, must be a power of 2
 * @param  num_threads number of pthreads to use; stack size set to 8MB per thread
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

// --- barrier -----------------------------------------------------------------

static void barrier_init(barrier_t *b, int total) {
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count      = 0;
    b->total      = total;
    b->generation = 0;
}

static void barrier_destroy(barrier_t *b) {
    pthread_mutex_destroy(&b->mutex);
    pthread_cond_destroy(&b->cond);
}

static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->mutex);

    int gen = b->generation;
    b->count++;

    if (b->count == b->total) {
        // last thread to arrive resets count and wakes all waiters
        b->count = 0;
        b->generation++;
        pthread_cond_broadcast(&b->cond);
    } else {
        // wait until generation advances, guarding against spurious wakeups
        while (b->generation == gen) {
            pthread_cond_wait(&b->cond, &b->mutex);
        }
    }

    pthread_mutex_unlock(&b->mutex);
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

static void *fft_thread(void *arg) {
    thread_arg_t *t      = (thread_arg_t *)arg;
    fft_shared_t *shared = t->shared;
    int           tid    = t->tid;
    int           nt     = shared->num_threads;

    for (int s = 0; s < shared->num_stages; s++) {
        int    len  = shared->stages[s].len;
        double ang  = shared->stages[s].ang;
        int    n    = shared->n;

        complex_t wlen = { cos(ang), sin(ang) };

        // each thread handles every nt-th group starting from tid
        for (int i = tid * len; i < n; i += nt * len) {
            complex_t w = { 1.0, 0.0 };     // twiddle factor, starts at w^0

            // butterfly between upper and lower halves of this group
            for (int j = 0; j < len / 2; j++) {
                complex_t u = shared->x[i + j];
                // multiply lower-half element by current twiddle factor
                complex_t v = {
                    shared->x[i + j + len/2].re * w.re - shared->x[i + j + len/2].im * w.im,
                    shared->x[i + j + len/2].re * w.im + shared->x[i + j + len/2].im * w.re
                };

                // combine into two output elements (butterfly operation)
                shared->x[i + j]         = (complex_t){ u.re + v.re, u.im + v.im };
                shared->x[i + j + len/2] = (complex_t){ u.re - v.re, u.im - v.im };

                // rotate twiddle factor by wlen for next butterfly pair
                double tmp = w.re * wlen.re - w.im * wlen.im;
                w.im       = w.re * wlen.im + w.im * wlen.re;
                w.re       = tmp;
            }
        }

        // all threads must finish this stage before any thread starts the next
        barrier_wait(t->barrier);
    }

    return NULL;
}

static void fft_parallel(complex_t *x, int n, int num_threads) {
    bit_reverse(x, n);

    // precompute all stage parameters so workers read from a fixed array;
    // eliminates any race between main updating shared state and workers reading it
    int num_stages = 0;
    for (int len = 2; len <= n; len <<= 1) { num_stages++; }

    stage_t *stages = malloc(num_stages * sizeof(stage_t));
    int s = 0;
    for (int len = 2; len <= n; len <<= 1) {
        stages[s].len = len;
        stages[s].ang = -2.0 * PI / len;
        s++;
    }

    fft_shared_t shared = { x, n, num_threads, num_stages, stages };
    barrier_t    barrier;
    barrier_init(&barrier, num_threads);   // workers only, main does not participate

    pthread_t    *threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t *args    = malloc(num_threads * sizeof(thread_arg_t));

    // set stack size explicitly to avoid overflow on large inputs with many threads
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);  // 8MB per thread

    for (int i = 0; i < num_threads; i++) {
        args[i].tid     = i;
        args[i].shared  = &shared;
        args[i].barrier = &barrier;
        pthread_create(&threads[i], &attr, fft_thread, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_attr_destroy(&attr);

    barrier_destroy(&barrier);
    free(stages);
    free(threads);
    free(args);
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

    printf("fft_pthreads n=%d threads=%d time_ms=%.4f\n", n, num_threads, elapsed_ms(t0, t1));

    // dump magnitudes to stdout when -v is passed (for testing only)
    if (verbose) {
        print_output(x, n);
    }

    free(x);
    return 0;
}