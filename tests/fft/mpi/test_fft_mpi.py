# file:        test_fft_mpi.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: validates fft_mpi output against fft_baseline across process counts and input types.
#              fft_mpi is a segmented fft — each rank processes n/nprocs samples independently,
#              so the comparison is per-chunk magnitude against the baseline run on the same chunk.
#              all tests run at 1, 2, 4, and 8 processes.

import subprocess
import struct
import tempfile
import os
import numpy as np

BASELINE_BIN = os.path.join(os.path.dirname(__file__), '../../../fft/baseline/fft_baseline')
MPI_BIN      = os.path.join(os.path.dirname(__file__), '../../../fft/mpi/fft_mpi')
PROC_COUNTS  = [1, 2, 4, 8]

ABS_TOL = 1e-3
REL_TOL = 1e-4


def main():
    tests = [
        test_pure_sine,
        test_all_zeros,
        test_impulse,
        test_multi_tone,
        test_random_signal,
        test_larger_input,
    ]

    print('=== fft_mpi correctness tests ===')
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

def test_pure_sine():
    # single frequency sine — verifies basic correctness across process counts
    n       = 1024
    t       = np.arange(n)
    samples = np.sin(2 * np.pi * 10 * t / n).astype(np.float32)
    run_against_baseline('pure sine (n=1024, bin=10)', samples)


def test_all_zeros():
    # zero input — all chunks should produce zero magnitudes
    n       = 1024
    samples = np.zeros(n, dtype=np.float32)
    run_against_baseline('all zeros (n=1024)', samples)


def test_impulse():
    # unit impulse at index 0 — only rank 0's chunk is non-trivial
    n          = 1024
    samples    = np.zeros(n, dtype=np.float32)
    samples[0] = 1.0
    run_against_baseline('unit impulse (n=1024)', samples)


def test_multi_tone():
    # two sine waves — checks correctness across chunk boundaries
    n       = 1024
    t       = np.arange(n)
    samples = (np.sin(2 * np.pi * 5  * t / n) +
               np.sin(2 * np.pi * 50 * t / n)).astype(np.float32)
    run_against_baseline('two-tone sine (n=1024, bins=5,50)', samples)


def test_random_signal():
    # random broadband signal — general correctness check
    n       = 2048
    rng     = np.random.default_rng(42)
    samples = rng.standard_normal(n).astype(np.float32)
    run_against_baseline('random signal (n=2048, seed=42)', samples)


def test_larger_input():
    # larger n to stress chunk distribution across more processes
    n       = 8192
    rng     = np.random.default_rng(7)
    samples = rng.standard_normal(n).astype(np.float32)
    run_against_baseline('random signal (n=8192, seed=7)', samples)


# helpers ---------------------------------------------------------------------

def run_against_baseline(label, samples):
    # for each process count, compare fft_mpi output against baseline run on same chunks
    n    = len(samples)
    path = write_binary(samples)
    try:
        for np_ in PROC_COUNTS:
            # skip if chunk size would not be a power of 2
            chunk = n // np_
            if chunk == 0 or (chunk & (chunk - 1)) != 0:
                print(f'  [SKIP] {label} nprocs={np_} (chunk={chunk} not power of 2)')
                continue
            expected = run_baseline_chunks(path, n, np_)
            got      = run_mpi(path, n, np_)
            check(f'{label} nprocs={np_}', got, expected)
    finally:
        os.unlink(path)


def write_binary(samples):
    # write float32 samples to a temp binary file, return the file path
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(struct.pack(f'{len(samples)}f', *samples))
    f.close()
    return f.name


def run_baseline_chunks(bin_path, n, nprocs):
    # run fft_baseline independently on each n/nprocs chunk of the input;
    # returns concatenated magnitudes matching the segmented layout of fft_mpi
    with open(bin_path, 'rb') as f:
        raw = f.read()
    all_samples = np.frombuffer(raw, dtype=np.float32)

    chunk    = n // nprocs
    all_mags = []

    for r in range(nprocs):
        chunk_samples = all_samples[r * chunk:(r + 1) * chunk]
        chunk_path    = write_binary(chunk_samples)
        try:
            result = subprocess.run(
                [BASELINE_BIN, chunk_path, str(chunk), '-v'],
                capture_output=True, text=True
            )
            assert result.returncode == 0, f'fft_baseline failed:\n{result.stderr}'
            lines = result.stdout.splitlines()
            mags  = [float(l) for l in lines[1:] if l.strip()]
            all_mags.extend(mags)
        finally:
            os.unlink(chunk_path)

    return np.array(all_mags)


def run_mpi(bin_path, n, nprocs):
    # run fft_mpi with -v and return output magnitudes as a numpy array;
    # first line of stdout is the timing summary, remaining lines are magnitudes
    result = subprocess.run(
        ['mpirun', '-np', str(nprocs), MPI_BIN, bin_path, str(n), '-v'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f'fft_mpi failed (nprocs={nprocs}):\n{result.stderr}'
    lines = result.stdout.splitlines()
    return np.array([float(l) for l in lines[1:] if l.strip()])


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