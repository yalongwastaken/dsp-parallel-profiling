# file:        test_fir_mpi.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: validates fir_mpi output against fir_baseline across process counts and input types.
#              fir_mpi scatters input with overlap so each rank can compute boundary samples
#              correctly; the full output is compared against a single baseline run on the
#              complete signal. all tests run at 1, 2, 4, and 8 processes.

import subprocess
import struct
import tempfile
import os
import numpy as np

BASELINE_BIN = os.path.join(os.path.dirname(__file__), '../../../fir/baseline/fir_baseline')
MPI_BIN      = os.path.join(os.path.dirname(__file__), '../../../fir/mpi/fir_mpi')
PROC_COUNTS  = [1, 2, 4, 8]

NUM_TAPS = 101
CUTOFF   = 0.1

ABS_TOL = 1e-5
REL_TOL = 1e-5


def main():
    tests = [
        test_low_frequency,
        test_high_frequency,
        test_all_zeros,
        test_impulse,
        test_random_signal,
        test_boundary_overlap,
    ]

    print('=== fir_mpi correctness tests ===')
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            pass

    total = len(tests) * len(PROC_COUNTS)
    print(f'\n{passed * len(PROC_COUNTS)}/{total} checks passed')


# tests -----------------------------------------------------------------------

def test_low_frequency():
    # low-frequency sine below cutoff — passes through; checks basic correctness
    n       = 1024
    t       = np.arange(n)
    samples = np.sin(2 * np.pi * 5 * t / n).astype(np.float32)
    run_against_baseline('low-freq sine (n=1024, bin=5)', samples)


def test_high_frequency():
    # high-frequency sine above cutoff — attenuated; checks all ranks agree
    n       = 1024
    t       = np.arange(n)
    samples = np.sin(2 * np.pi * 400 * t / n).astype(np.float32)
    run_against_baseline('high-freq sine (n=1024, bin=400)', samples)


def test_all_zeros():
    # zero input — output should be zero regardless of process count
    n       = 1024
    samples = np.zeros(n, dtype=np.float32)
    run_against_baseline('all zeros (n=1024)', samples)


def test_impulse():
    # unit impulse — output is the filter's impulse response;
    # critical test for overlap correctness at rank boundaries
    n          = 1024
    samples    = np.zeros(n, dtype=np.float32)
    samples[0] = 1.0
    run_against_baseline('unit impulse (n=1024)', samples)


def test_random_signal():
    # random broadband signal — general correctness check
    n       = 2048
    rng     = np.random.default_rng(42)
    samples = rng.standard_normal(n).astype(np.float32)
    run_against_baseline('random signal (n=2048, seed=42)', samples)


def test_boundary_overlap():
    # signal with energy near rank boundaries — directly stresses the overlap logic
    n       = 4096
    rng     = np.random.default_rng(99)
    samples = rng.standard_normal(n).astype(np.float32)
    run_against_baseline('boundary overlap stress (n=4096, seed=99)', samples)


# helpers ---------------------------------------------------------------------

def run_against_baseline(label, samples, num_taps=NUM_TAPS, cutoff=CUTOFF):
    # run baseline once on full signal, then compare mpi output at each process count
    n    = len(samples)
    path = write_binary(samples)
    try:
        expected = run_baseline(path, n, num_taps, cutoff)
        for np_ in PROC_COUNTS:
            if n % np_ != 0:
                print(f'  [SKIP] {label} nprocs={np_} (n={n} not divisible by {np_})')
                continue
            got = run_mpi(path, n, np_, num_taps, cutoff)
            check(f'{label} nprocs={np_}', got, expected)
    finally:
        os.unlink(path)


def write_binary(samples):
    # write float32 samples to a temp binary file, return the file path
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(struct.pack(f'{len(samples)}f', *samples))
    f.close()
    return f.name


def run_baseline(bin_path, n, num_taps=NUM_TAPS, cutoff=CUTOFF):
    # run fir_baseline with -v on the full signal and return output samples;
    # first line of stdout is the timing summary, remaining lines are samples
    result = subprocess.run(
        [BASELINE_BIN, bin_path, str(n), str(num_taps), str(cutoff), '-v'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f'fir_baseline failed:\n{result.stderr}'
    lines = result.stdout.splitlines()
    return np.array([float(l) for l in lines[1:] if l.strip()], dtype=np.float32)


def run_mpi(bin_path, n, nprocs, num_taps=NUM_TAPS, cutoff=CUTOFF):
    # run fir_mpi with -v and return output samples as a numpy array;
    # first line of stdout is the timing summary, remaining lines are samples
    result = subprocess.run(
        ['mpirun', '-np', str(nprocs), MPI_BIN, bin_path, str(n),
         str(num_taps), str(cutoff), '-v'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f'fir_mpi failed (nprocs={nprocs}):\n{result.stderr}'
    lines = result.stdout.splitlines()
    return np.array([float(l) for l in lines[1:] if l.strip()], dtype=np.float32)


def check(label, got, expected):
    # assert got and expected agree within tolerance, print pass/fail
    close  = np.allclose(got, expected, atol=ABS_TOL, rtol=REL_TOL)
    status = 'PASS' if close else 'FAIL'
    print(f'  [{status}] {label}')
    if not close:
        worst = np.max(np.abs(got - expected))
        print(f'         max absolute error: {worst:.6e}')
    assert close, f'test failed: {label}'


if __name__ == '__main__':
    main()