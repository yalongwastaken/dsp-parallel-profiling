# EECE5640 Final Project — FFT & FIR Filter Parallelization Study
**Team:** Vaidehi Gohil, Anthony Yalong
**Course:** EECE5640
**Platform:** Explorer Cluster (SLURM/SBATCH)

## Overview
This repository contains parallel implementations of two DSP workloads:
- **Fast Fourier Transform (FFT)**
- **FIR Filtering (Finite Impulse Response)**

Each workload is implemented using three parallelization middlewares:
- **Pthreads**
- **OpenMP**
- **OpenMPI** (extra credit)

The goal is to benchmark and analyze parallel scaling behavior (speedup, efficiency) across middlewares and input sizes on a single multi-core node.

## Repo Structure
```
.
├── administrative/            # documents, reports, and write-ups
├── data/                      # all input datasets (see Data section)
├── fft/
│   ├── baseline/              # serial baseline (fft_baseline.c)
│   ├── mpi/
│   ├── openmp/
│   └── pthreads/
├── fir/
│   ├── baseline/              # serial baseline (fir_baseline.c)
│   ├── mpi/
│   ├── openmp/
│   └── pthreads/
├── tests/
│   ├── fft/                   # correctness tests for fft implementations
│   └── fir/                   # correctness tests for fir implementations
├── results/                   # job output logs (gitignored)
├── scripts/
│   ├── generate_input.py      # generates synthetic input binaries
│   ├── submit_baseline.sh     # submits baseline jobs
│   ├── submit_mpi.sh          # submits MPI jobs for all process counts
│   ├── submit_openmp.sh       # submits OpenMP jobs for all thread counts
│   └── submit_pthreads.sh     # submits Pthreads jobs for all thread counts
├── .gitignore
└── README.md
```

## Data
| Dataset | Size | Source |
|---------|------|--------|
| `input_small.bin` | 2^20 samples | Generated via `generate_input.py` |
| `input_medium.bin` | 2^24 samples | Generated via `generate_input.py` |
| `input_large.bin` | 2^26 samples | Generated via `generate_input.py` |

The `data/` directory is gitignored for large files. Run `generate_input.py` before executing any jobs.

```bash
python3 scripts/generate_input.py
```

## Building

Each workload subdirectory has its own Makefile. From the repo root:

```bash
cd fft && make && cd ../fir && make && cd ..
```

## Testing

Two levels of correctness testing are used:

**Baseline tests** compare baseline output against a Python reference implementation that mirrors the C normalization exactly.

```bash
python3 tests/fft/baseline/test_fft_baseline.py
python3 tests/fir/baseline/test_fir_baseline.py
```

**Parallel tests** (Pthreads, OpenMP, MPI) compare each parallel implementation's output directly against the baseline binary using `-v`, isolating implementation bugs from any reference discrepancy.

```bash
python3 tests/fft/pthreads/test_fft_pthreads.py
python3 tests/fft/openmp/test_fft_openmp.py
python3 tests/fft/mpi/test_fft_mpi.py
# and equivalents under tests/fir/
```

All tests use small synthetic inputs generated internally and do not depend on the `data/` directory. Requires `numpy` (`pip install numpy`).

## Running

```bash
# fft baseline
./fft/baseline/fft_baseline data/input_small.bin 1048576

# fir baseline
./fir/baseline/fir_baseline data/input_small.bin 1048576 101 0.1
```

Pass `-v` to dump output samples to stdout (for testing only):

```bash
./fft/baseline/fft_baseline data/input_small.bin 1048576 -v
./fir/baseline/fir_baseline data/input_small.bin 1048576 101 0.1 -v
```

## Measurements

For each configuration we collect:
- Wall-clock execution time (5 runs, report mean ± std dev)
- Speedup: `S(p) = T(1) / T(p)`
- Parallel efficiency: `E(p) = S(p) / p`

Thread/process counts tested: **1, 2, 4, 8, 16, 32**