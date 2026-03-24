/**
 * @file    fir_mpi.c
 * @author  Vaidehi Gohil, Anthony Yalong
 * @brief   mpi parallel fir low-pass filter (hamming window) — rank 0 scatters
 *          input chunks with overlap for the filter window, each rank computes
 *          its output samples independently, rank 0 gathers results.
 * @usage   mpirun -np <nprocs> ./fir_mpi <input.bin> <n_samples> <num_taps> <cutoff_ratio> [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define PI 3.14159265358979323846

// --- prototypes --------------------------------------------------------------

/**
 * @brief  generate fir low-pass filter coefficients using the hamming window method;
 *         caller is responsible for freeing the returned array
 * @param  num_taps     number of filter taps, must be odd for linear phase
 * @param  cutoff_ratio normalized cutoff frequency in (0, 0.5), i.e. fc / fs
 * @return heap-allocated array of num_taps float coefficients, or NULL on failure
 */
static float *hamming_lowpass(int num_taps, double cutoff_ratio);

/**
 * @brief  apply direct-form fir filter to local input chunk, writing chunk_out
 *         output samples; input must include num_taps-1 overlap samples prepended
 *         so that boundary output samples are computed correctly
 * @param  in        local input buffer, length chunk_out + num_taps - 1
 * @param  out       local output buffer, length chunk_out
 * @param  chunk_out number of output samples this rank produces
 * @param  h         filter coefficient array, length num_taps
 * @param  num_taps  number of filter taps
 */
static void fir_filter(const float *in, float *out, int chunk_out,
                       const float *h, int num_taps);

/**
 * @brief  print filtered output samples to stdout, one per line
 * @param  out  output sample array, length n
 * @param  n    number of samples
 */
static void print_output(const float *out, int n);

// --- filter design -----------------------------------------------------------

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

static void fir_filter(const float *in, float *out, int chunk_out,
                       const float *h, int num_taps) {
    // in is pre-padded with num_taps-1 overlap samples so index i+num_taps-1
    // in the padded buffer corresponds to output sample i
    for (int i = 0; i < chunk_out; i++) {
        double acc = 0.0;
        for (int k = 0; k < num_taps; k++) {
            // padded buffer offset ensures no out-of-bounds access
            acc += h[k] * in[i + num_taps - 1 - k];
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5 || argc > 6) {
        if (rank == 0) {
            fprintf(stderr,
                "usage: mpirun -np <nprocs> %s <input.bin> <n_samples> <num_taps> <cutoff_ratio> [-v]\n",
                argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *path = argv[1];
    int n            = atoi(argv[2]);
    int num_taps     = atoi(argv[3]);
    double cutoff    = atof(argv[4]);
    int verbose      = (argc == 6 && strcmp(argv[5], "-v") == 0);

    if (n <= 0 || num_taps <= 0 || cutoff <= 0.0 || cutoff >= 0.5) {
        if (rank == 0) {
            fprintf(stderr, "error: n>0, num_taps>0, cutoff_ratio in (0, 0.5)\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (num_taps % 2 == 0) {
        if (rank == 0) {
            fprintf(stderr, "error: num_taps should be odd for symmetric filter\n");
        }
        MPI_Finalize();
        return 1;
    }

    int chunk_out = n / size;   // output samples per rank
    int overlap   = num_taps - 1; // input samples needed from previous rank's region

    float *all_input  = NULL;
    float *all_output = NULL;

    // rank 0 reads full input and prepares padded scatter buffer
    // each rank receives chunk_out + overlap samples so boundary convolutions
    // can be computed without inter-rank communication
    int send_count = chunk_out + overlap;
    float *send_buf = NULL;

    if (rank == 0) {
        all_input = malloc(n * sizeof(float));
        if (!all_input) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

        FILE *f = fopen(path, "rb");
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if ((int)fread(all_input, sizeof(float), n, f) != n) {
            fprintf(stderr, "error: expected %d float32 samples\n", n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(f);

        all_output = malloc(n * sizeof(float));
        if (!all_output) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

        // build contiguous send buffer: for each rank, prepend overlap zeros
        // (for rank 0) or overlap samples from previous rank's region
        send_buf = malloc(size * send_count * sizeof(float));
        if (!send_buf) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

        for (int r = 0; r < size; r++) {
            float *dst = send_buf + r * send_count;
            int    src_start = r * chunk_out - overlap;

            for (int i = 0; i < send_count; i++) {
                int idx = src_start + i;
                // zero-pad samples before start of input array
                dst[i] = (idx >= 0) ? all_input[idx] : 0.0f;
            }
        }
    }

    // each rank receives its padded input chunk
    float *local_in = malloc(send_count * sizeof(float));
    if (!local_in) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Scatter(send_buf, send_count, MPI_FLOAT, local_in, send_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float *local_out = malloc(chunk_out * sizeof(float));
    if (!local_out) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // design filter; all ranks compute identical coefficients
    float *h = hamming_lowpass(num_taps, cutoff);
    if (!h) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    // --- timed region --------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    fir_filter(local_in, local_out, chunk_out, h, num_taps);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    // -------------------------------------------------------------------------

    // gather output chunks back to rank 0
    MPI_Gather(local_out, chunk_out, MPI_FLOAT, all_output, chunk_out, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("fir_mpi n=%d taps=%d nprocs=%d time_ms=%.4f\n",
               n, num_taps, size, (t_end - t_start) * 1e3);

        // dump output samples to stdout when -v is passed (for testing only)
        if (verbose) {
            print_output(all_output, n);
        }

        free(all_input);
        free(all_output);
        free(send_buf);
    }

    free(local_in);
    free(local_out);
    free(h);
    MPI_Finalize();
    return 0;
}