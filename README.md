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
- **OpenMPI**

The goal is to benchmark and analyze parallel scaling behavior (speedup, efficiency) across middlewares and input sizes on a single multi-core node.

## Repo Structure

```
.
├── administrative/            # documents, reports, and write-ups
├── data/                      # all input datasets (see Data section)
├── fft/
│   ├── mpi/
│   ├── openmp/
│   └── pthreads/
├── fir/
│   ├── mpi/
│   ├── openmp/
│   └── pthreads/
├── results/                   # job output logs (gitignored)
├── scripts/
│   ├── generate_input.py      # generates synthetic input CSVs
│   ├── submit_mpi.sh          # submits MPI jobs for all process counts 
│   └── submit_omp.sh          # submits OpenMP/Pthreads jobs for all thread counts
├── .gitignore
└── README.md
```

## Data

| Dataset | Size | Source |
|---------|------|--------|
| `input_small.csv` | 2^16 samples | Generated via `generate_input.py` |
| `input_medium.csv` | 2^20 samples | Generated via `generate_input.py` |
| `input_large.csv` | ~4M samples | [LibriSpeech](https://www.openslr.org/12) — download manually and convert |

The `data/` directory is gitignored for large files. Ensure to generate or download inputs independently before running jobs.


## Measurements

For each configuration we collect:
- Wall-clock execution time (5 runs, report mean ± std dev)
- Speedup: `S(p) = T(1) / T(p)`
- Parallel efficiency: `E(p) = S(p) / p`

Thread/process counts tested: **1, 2, 4, 8, 16, 32**