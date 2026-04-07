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

The goal is to benchmark and analyze parallel scaling behavior (speedup, efficiency) across middlewares and input sizes on a single multi-core node. Benchmarks are run on both synthetic and real-world (LibriSpeech) input data to compare performance across signal types.

## Repo Structure
```
.
├── administrative/            # documents, reports, and write-ups
├── data/                      # input datasets (gitignored except README)
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
├── results/                   # benchmark output files
├── scripts/
│   ├── generate_input.py      # generates synthetic float32 input binaries
│   ├── download_dataset.py    # downloads and processes librispeech input binaries
│   ├── submit_generate.sh     # sbatch wrapper for generate_input.py
│   ├── submit_download.sh     # sbatch wrapper for download_dataset.py
│   ├── submit_baseline.sh     # submits baseline benchmark jobs
│   ├── submit_pthreads.sh     # submits pthreads benchmark jobs
│   ├── submit_openmp.sh       # submits openmp benchmark jobs
│   └── submit_mpi.sh          # submits mpi benchmark jobs
├── .gitignore
├── requirements.txt
└── README.md
```

## Data

Both input pipelines produce three dataset sizes at 16kHz float32:

| Dataset | Samples | Size |
|---------|---------|------|
| small | 2^20 | ~4 MB |
| medium | 2^24 | ~64 MB |
| large | 2^26 | ~256 MB |

**Synthetic** inputs are multi-tone sine waves with Gaussian noise, generated via `generate_input.py`. Output files are named `input_{size}_generated.bin`.

**Real-world** inputs are sourced from the [LibriSpeech](https://www.openslr.org/12) dev-clean corpus, resampled to 16kHz and trimmed/tiled to exact sample counts via `download_dataset.py`. Output files are named `input_{size}_downloaded.bin`.

The `data/` directory is gitignored due to file size. Generate or download inputs before running any jobs.

## Setup

### Local (macOS)

```bash
# install homebrew gcc for openmp support
brew install gcc                   # provides gcc-15

# clone repo
git clone https://github.com/yalongwastaken/eece5640-finalproject.git
cd eece5640-finalproject

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install python dependencies
pip install -r requirements.txt
```

### Explorer

```bash
# clone repo
git clone https://github.com/yalongwastaken/eece5640-finalproject.git
cd eece5640-finalproject

# load required modules
module load OpenMPI/4.1.6
module load python/3.13.5

# activate virtual environment (numpy and soundfile must be installed)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generating Input Data

**On Explorer (recommended — avoids daemon killing long processes):**
```bash
sbatch scripts/submit_generate.sh   # synthetic data
sbatch scripts/submit_download.sh   # librispeech data
```

**Locally:**
```bash
python3 scripts/generate_input.py
python3 scripts/download_dataset.py
```

Both scripts write to `data/` and will overwrite existing files.

## Building

MPI targets require `mpicc` — load the OpenMPI module first. On macOS, `CC` is set to `gcc-15` in both Makefiles for OpenMP support; change this if your Homebrew GCC version differs.

```bash
cd fft && make && cd ../fir && make && cd ..
```

Submission scripts on Explorer automatically build before running benchmarks.

## Testing

**Baseline tests** compare baseline output against a Python reference implementation.
```bash
python3 tests/fft/baseline/test_fft_baseline.py
python3 tests/fir/baseline/test_fir_baseline.py
```

**Parallel tests** compare each parallel implementation's output directly against the baseline binary, isolating implementation bugs from reference discrepancy.
```bash
python3 tests/fft/pthreads/test_fft_pthreads.py
python3 tests/fft/openmp/test_fft_openmp.py
python3 tests/fft/mpi/test_fft_mpi.py
python3 tests/fir/pthreads/test_fir_pthreads.py
python3 tests/fir/openmp/test_fir_openmp.py
python3 tests/fir/mpi/test_fir_mpi.py
```

All tests generate their own small inputs internally and do not depend on the `data/` directory.

## Running Manually

The `-v` flag dumps output samples to stdout for testing. Omit it for clean benchmark runs.

```bash
# baseline
./fft/baseline/fft_baseline <input.bin> <n_samples> [-v]
./fir/baseline/fir_baseline <input.bin> <n_samples> <num_taps> <cutoff> [-v]

# pthreads
./fft/pthreads/fft_pthreads <input.bin> <n_samples> <num_threads> [-v]
./fir/pthreads/fir_pthreads <input.bin> <n_samples> <num_taps> <cutoff> <num_threads> [-v]

# openmp
./fft/openmp/fft_openmp <input.bin> <n_samples> <num_threads> [-v]
./fir/openmp/fir_openmp <input.bin> <n_samples> <num_taps> <cutoff> <num_threads> [-v]

# mpi
mpirun -np <nprocs> ./fft/mpi/fft_mpi <input.bin> <n_samples> [-v]
mpirun -np <nprocs> ./fir/mpi/fir_mpi <input.bin> <n_samples> <num_taps> <cutoff> [-v]
```

## Submitting Jobs on Explorer

Each submission script runs both FFT and FIR across all thread/process counts and all six datasets (3 sizes × 2 input types). Make sure `results/` exists and both input pipelines have been run before submitting.

```bash
sbatch scripts/submit_baseline.sh
sbatch scripts/submit_pthreads.sh
sbatch scripts/submit_openmp.sh
sbatch scripts/submit_mpi.sh
```

Monitor and retrieve results:
```bash
squeue -u $USER
cat results/<middleware>_<jobid>.out
```

## Measurements

For each configuration we collect:
- Wall-clock execution time (5 runs, report mean ± std dev)
- Speedup: `S(p) = T(1) / T(p)`
- Parallel efficiency: `E(p) = S(p) / p`

Thread/process counts tested: **1, 2, 4, 8, 16, 32**
Input types: **synthetic (generated)** and **real-world (librispeech)**