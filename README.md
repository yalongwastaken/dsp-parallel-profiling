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
│   ├── mpi/                   # mpi implementation (fft_mpi.c)
│   ├── openmp/                # openmp implementation (fft_openmp.c)
│   ├── pthreads/              # pthreads implementation (fft_pthreads.c)
│   └── Makefile
├── fir/
│   ├── baseline/              # serial baseline (fir_baseline.c)
│   ├── mpi/                   # mpi implementation (fir_mpi.c)
│   ├── openmp/                # openmp implementation (fir_openmp.c)
│   ├── pthreads/              # pthreads implementation (fir_pthreads.c)
│   └── Makefile
├── tests/
│   ├── fft/
│   │   ├── baseline/          # correctness tests vs python reference
│   │   ├── pthreads/          # correctness tests vs baseline binary
│   │   ├── openmp/            # correctness tests vs baseline binary
│   │   └── mpi/               # correctness tests vs baseline binary
│   └── fir/
│       ├── baseline/          # correctness tests vs python reference
│       ├── pthreads/          # correctness tests vs baseline binary
│       ├── openmp/            # correctness tests vs baseline binary
│       └── mpi/               # correctness tests vs baseline binary
├── results/                   # job output logs (gitignored)
├── scripts/
│   ├── generate_input.py      # generates synthetic input binaries
│   ├── submit_generate.sh     # submits input generation as sbatch job
│   ├── submit_baseline.sh     # submits baseline jobs
│   ├── submit_pthreads.sh     # submits pthreads jobs for all thread counts
│   ├── submit_openmp.sh       # submits openmp jobs for all thread counts
│   └── submit_mpi.sh          # submits mpi jobs for all process counts
├── .gitignore
└── README.md
```

## Dependencies

**On Explorer:**
```bash
module load OpenMPI/4.1.6
module load python/3.13.5
source .venv/bin/activate      # numpy must be available
```

## Data

| Dataset | Size | Source |
|---------|------|--------|
| `input_small.bin` | 2^20 samples | Generated via `generate_input.py` |
| `input_medium.bin` | 2^24 samples | Generated via `generate_input.py` |
| `input_large.bin` | 2^26 samples | Generated via `generate_input.py` |

The `data/` directory is gitignored. Generate inputs before running any jobs.

**On Explorer (recommended — avoids daemon killing long-running processes):**
```bash
sbatch scripts/submit_generate.sh
```

**Locally:**
```bash
python3 scripts/generate_input.py
```

## Building

Each workload subdirectory has its own Makefile. MPI targets require `mpicc` — load the module first.

```bash
module load OpenMPI/4.1.6
cd fft && make && cd ../fir && make && cd ..
```

## Testing

Two levels of correctness testing are used:

**Baseline tests** compare baseline output against a Python reference implementation that mirrors the C normalization exactly.
```bash
python3 tests/fft/baseline/test_fft_baseline.py
python3 tests/fir/baseline/test_fir_baseline.py
```

**Parallel tests** compare each parallel implementation's output directly against the baseline binary using `-v`, isolating implementation bugs from any reference discrepancy.
```bash
python3 tests/fft/pthreads/test_fft_pthreads.py
python3 tests/fft/openmp/test_fft_openmp.py
python3 tests/fft/mpi/test_fft_mpi.py
python3 tests/fir/pthreads/test_fir_pthreads.py
python3 tests/fir/openmp/test_fir_openmp.py
python3 tests/fir/mpi/test_fir_mpi.py
```

All tests use small synthetic inputs generated internally and do not depend on the `data/` directory.

## Running Manually

```bash
# baseline
./fft/baseline/fft_baseline data/input_small.bin 1048576
./fir/baseline/fir_baseline data/input_small.bin 1048576 101 0.1

# pthreads
./fft/pthreads/fft_pthreads data/input_small.bin 1048576 <num_threads>
./fir/pthreads/fir_pthreads data/input_small.bin 1048576 101 0.1 <num_threads>

# openmp
./fft/openmp/fft_openmp data/input_small.bin 1048576 <num_threads>
./fir/openmp/fir_openmp data/input_small.bin 1048576 101 0.1 <num_threads>

# mpi
mpirun -np <nprocs> ./fft/mpi/fft_mpi data/input_small.bin 1048576
mpirun -np <nprocs> ./fir/mpi/fir_mpi data/input_small.bin 1048576 101 0.1
```

Pass `-v` to dump output samples to stdout (for testing only):
```bash
./fft/baseline/fft_baseline data/input_small.bin 1048576 -v
```

## Submitting Jobs on Explorer

Make sure `results/` exists before submitting:
```bash
mkdir -p results
```

Then submit each middleware:
```bash
sbatch scripts/submit_baseline.sh
sbatch scripts/submit_pthreads.sh
sbatch scripts/submit_openmp.sh
sbatch scripts/submit_mpi.sh
```

Monitor jobs:
```bash
squeue -u $USER
```

View output:
```bash
cat results/baseline_<jobid>.out
cat results/pthreads_<jobid>.out
cat results/openmp_<jobid>.out
cat results/mpi_<jobid>.out
```

## Measurements

For each configuration we collect:
- Wall-clock execution time (5 runs, report mean ± std dev)
- Speedup: `S(p) = T(1) / T(p)`
- Parallel efficiency: `E(p) = S(p) / p`

Thread/process counts tested: **1, 2, 4, 8, 16, 32**