/**
 * @file    fft_mpi.c
 * @author  Vaidehi Gohil, Anthony Yalong
 * @brief   mpi parallel fft — rank 0 scatters signal chunks, each rank computes
 *          a local cooley-tukey fft on its chunk, rank 0 gathers results.
 *          note: this is a segmented fft; each rank processes n/p samples
 *          independently, so frequency resolution scales with process count.
 * @usage   mpirun -np <nprocs> ./fft_mpi <input.bin> <n_samples> [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define PI 3.14159265358979323846

typedef struct { double re, im; } complex_t;

// --- prototypes --------------------------------------------------------------

/**
 * @brief  reorder elements of x into bit-reversed index order in-place;
 *         called before butterfly passes on each rank's local chunk
 * @param  x  array of complex samples, length n
 * @param  n  number of samples, must be a power of 2
 */
static void bit_reverse(complex_t *x, int n);

/**
 * @brief  compute the in-place iterative cooley-tukey fft of x
 * @param  x  array of complex samples, length n; overwritten with frequency-domain output
 * @param  n  number of samples, must be a power of 2
 */
static void fft(complex_t *x, int n);

/**
 * @brief  print fft output magnitudes to stdout, one per line
 * @param  x  frequency-domain complex array, length n
 * @param  n  number of samples
 */
static void print_output(const complex_t *x, int n);

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

static void fft(complex_t *x, int n) {
    bit_reverse(x, n);

    // each pass doubles butterfly width, merging sub-transforms bottom-up
    for (int len = 2; len <= n; len <<= 1) {
        // twiddle factor base for this stage: e^(-2*pi*i / len)
        double    ang  = -2.0 * PI / len;
        complex_t wlen = { cos(ang), sin(ang) };

        // process each group of len elements
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 || argc > 4) {
        if (rank == 0) {
            fprintf(stderr, "usage: mpirun -np <nprocs> %s <input.bin> <n_samples> [-v]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *path = argv[1];
    int n            = atoi(argv[2]);
    int verbose      = (argc == 4 && strcmp(argv[3], "-v") == 0);

    // n must be divisible by number of ranks and each chunk must be power of 2
    int chunk = n / size;
    if (n <= 0 || (n & (n - 1)) != 0 || (chunk & (chunk - 1)) != 0) {
        if (rank == 0) {
            fprintf(stderr, "error: n_samples and n_samples/nprocs must both be powers of 2\n");
        }
        MPI_Finalize();
        return 1;
    }

    float     *all_buf    = NULL;
    complex_t *all_output = NULL;

    // rank 0 reads full input and allocates gather buffer
    if (rank == 0) {
        all_buf = malloc(n * sizeof(float));
        if (!all_buf) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

        FILE *f = fopen(path, "rb");
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if ((int)fread(all_buf, sizeof(float), n, f) != n) {
            fprintf(stderr, "error: expected %d float32 samples\n", n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(f);

        all_output = malloc(n * sizeof(complex_t));
        if (!all_output) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // scatter float32 chunks to all ranks
    float *local_buf = malloc(chunk * sizeof(float));
    if (!local_buf) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Scatter(all_buf, chunk, MPI_FLOAT, local_buf, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // promote local float32 chunk to double complex
    complex_t *local_x = malloc(chunk * sizeof(complex_t));
    if (!local_x) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int i = 0; i < chunk; i++) {
        local_x[i] = (complex_t){ local_buf[i], 0.0 };
    }
    free(local_buf);

    // --- timed region --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    fft(local_x, chunk);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    // -------------------------------------------------------------------------

    // gather all local frequency-domain chunks back to rank 0;
    // MPI_Gather requires a contiguous buffer so we define a custom type
    MPI_Datatype mpi_complex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_complex);
    MPI_Type_commit(&mpi_complex);

    MPI_Gather(local_x, chunk, mpi_complex, all_output, chunk, mpi_complex, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_complex);

    if (rank == 0) {
        printf("fft_mpi n=%d nprocs=%d chunk=%d time_ms=%.4f\n",
               n, size, chunk, (t_end - t_start) * 1e3);

        // dump magnitudes to stdout when -v is passed (for testing only)
        if (verbose) {
            print_output(all_output, n);
        }

        free(all_buf);
        free(all_output);
    }

    free(local_x);
    MPI_Finalize();
    return 0;
}