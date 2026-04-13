# file:        test_fft_openmp.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: validates fft_openmp output against fft_baseline across thread counts and input types.
#              all tests run at 1, 2, 4, and 8 threads.

import subprocess
import struct
import tempfile
import os
import numpy as np

BASELINE_BIN  = os.path.join(os.path.dirname(__file__), '../../../fft/baseline/fft_baseline')
OPENMP_BIN    = os.path.join(os.path.dirname(__file__), '../../../fft/openmp/fft_openmp')
THREAD_COUNTS = [1, 2, 4, 8]

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

    print('=== fft_openmp correctness tests ===')
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            pass

    total = len(tests) * len(THREAD_COUNTS)
    print(f'\n{passed * len(THREAD_COUNTS)}/{total} checks passed')


# tests -----------------------------------------------------------------------

def test_pure_sine():
    # single frequency sine — verifies basic correctness across thread counts
    n       = 1024
    t       = np.arange(n)
    samples = np.sin(2 * np.pi * 10 * t / n).astype(np.float32)
    run_against_baseline('pure sine (n=1024, bin=10)', samples)


def test_all_zeros():
    # zero input — fft of zeros should be zeros regardless of thread count
    n       = 256
    samples = np.zeros(n, dtype=np.float32)
    run_against_baseline('all zeros (n=256)', samples)


def test_impulse():
    # unit impulse — flat spectrum; sensitive to any indexing bugs
    n          = 256
    samples    = np.zeros(n, dtype=np.float32)
    samples[0] = 1.0
    run_against_baseline('unit impulse (n=256)', samples)


def test_multi_tone():
    # two sine waves — checks that independent frequency bins are unaffected by thread boundaries
    n       = 1024
    t       = np.arange(n)
    samples = (np.sin(2 * np.pi * 5  * t / n) +
               np.sin(2 * np.pi * 50 * t / n)).astype(np.float32)
    run_against_baseline('two-tone sine (n=1024, bins=5,50)', samples)


def test_random_signal():
    # random broadband signal — general correctness across all frequency bins
    n       = 2048
    rng     = np.random.default_rng(42)
    samples = rng.standard_normal(n).astype(np.float32)
    run_against_baseline('random signal (n=2048, seed=42)', samples)


def test_larger_input():
    # larger n to stress thread work distribution across more butterfly stages
    n       = 8192
    rng     = np.random.default_rng(7)
    samples = rng.standard_normal(n).astype(np.float32)
    run_against_baseline('random signal (n=8192, seed=7)', samples)


# helpers ---------------------------------------------------------------------

def run_against_baseline(label, samples):
    # run baseline once, then compare openmp output at each thread count
    n    = len(samples)
    path = write_binary(samples)
    try:
        expected = run_baseline(path, n)
        for nt in THREAD_COUNTS:
            got = run_openmp(path, n, nt)
            check(f'{label} threads={nt}', got, expected)
    finally:
        os.unlink(path)


def write_binary(samples):
    # write float32 samples to a temp binary file, return the file path
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(struct.pack(f'{len(samples)}f', *samples))
    f.close()
    return f.name


def run_baseline(bin_path, n):
    # run fft_baseline with -v and return output magnitudes as a numpy array;
    # first line of stdout is the timing summary, remaining lines are magnitudes
    result = subprocess.run(
        [BASELINE_BIN, bin_path, str(n), '-v'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f'fft_baseline failed:\n{result.stderr}'
    lines = result.stdout.splitlines()
    return np.array([float(l) for l in lines[1:] if l.strip()])


def run_openmp(bin_path, n, num_threads):
    # run fft_openmp with -v and return output magnitudes as a numpy array;
    # first line of stdout is the timing summary, remaining lines are magnitudes
    result = subprocess.run(
        [OPENMP_BIN, bin_path, str(n), str(num_threads), '-v'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f'fft_openmp failed (threads={num_threads}):\n{result.stderr}'
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