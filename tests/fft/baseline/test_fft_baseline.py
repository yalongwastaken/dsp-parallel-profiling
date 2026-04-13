# file:        test_fft_baseline.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: runs correctness tests for fft_baseline against numpy.fft as reference

import subprocess
import struct
import tempfile
import os
import numpy as np

# path to compiled binary relative to this file
FFT_BIN = os.path.join(os.path.dirname(__file__), '../../../fft/baseline/fft_baseline')

# tolerance for comparing magnitudes against numpy reference
ABS_TOL = 1e-3
REL_TOL = 1e-4


def main():
    tests = [
        test_pure_sine,
        test_all_zeros,
        test_impulse,
        test_dc_signal,
        test_multi_tone,
        test_random_signal,
    ]

    print('=== fft_baseline correctness tests ===')
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            pass

    print(f'\n{passed}/{len(tests)} tests passed')


# tests -----------------------------------------------------------------------

def test_pure_sine():
    # single frequency sine — fft should show a clear spike at that bin
    n        = 1024
    freq_bin = 10
    t        = np.arange(n)
    samples  = np.sin(2 * np.pi * freq_bin * t / n).astype(np.float32)

    path = write_binary(samples)
    try:
        got      = run_fft_baseline(path, n)
        expected = reference_magnitudes(samples)
        check('pure sine wave (n=1024, bin=10)', got, expected)
    finally:
        os.unlink(path)


def test_all_zeros():
    # zero input — fft of zeros is zeros
    n       = 256
    samples = np.zeros(n, dtype=np.float32)

    path = write_binary(samples)
    try:
        got      = run_fft_baseline(path, n)
        expected = reference_magnitudes(samples)
        check('all zeros (n=256)', got, expected)
    finally:
        os.unlink(path)


def test_impulse():
    # unit impulse at index 0 — fft should be flat (magnitude 1 everywhere)
    n          = 256
    samples    = np.zeros(n, dtype=np.float32)
    samples[0] = 1.0

    path = write_binary(samples)
    try:
        got      = run_fft_baseline(path, n)
        expected = reference_magnitudes(samples)
        check('unit impulse at index 0 (n=256)', got, expected)
    finally:
        os.unlink(path)


def test_dc_signal():
    # constant signal — all energy should be at bin 0
    n       = 512
    samples = np.ones(n, dtype=np.float32) * 3.0

    path = write_binary(samples)
    try:
        got      = run_fft_baseline(path, n)
        expected = reference_magnitudes(samples)
        check('dc signal (n=512, value=3.0)', got, expected)
    finally:
        os.unlink(path)


def test_multi_tone():
    # sum of two sine waves — fft should show spikes at both frequencies
    n       = 1024
    t       = np.arange(n)
    samples = (np.sin(2 * np.pi * 5  * t / n) +
               np.sin(2 * np.pi * 50 * t / n)).astype(np.float32)

    path = write_binary(samples)
    try:
        got      = run_fft_baseline(path, n)
        expected = reference_magnitudes(samples)
        check('two-tone sine (n=1024, bins=5,50)', got, expected)
    finally:
        os.unlink(path)


def test_random_signal():
    # random signal — broad spectrum, tests general correctness
    n       = 2048
    rng     = np.random.default_rng(42)
    samples = rng.standard_normal(n).astype(np.float32)

    path = write_binary(samples)
    try:
        got      = run_fft_baseline(path, n)
        expected = reference_magnitudes(samples)
        check('random signal (n=2048, seed=42)', got, expected)
    finally:
        os.unlink(path)


# helpers ---------------------------------------------------------------------

def write_binary(samples):
    # write float32 samples to a temp binary file, return the file path
    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
    f.write(struct.pack(f'{len(samples)}f', *samples))
    f.close()
    return f.name


def run_fft_baseline(bin_path, n):
    # run fft_baseline with -v and parse magnitude output from stdout;
    # first line is the timing summary, remaining lines are one magnitude per bin
    result = subprocess.run(
        [FFT_BIN, bin_path, str(n), '-v'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f'fft_baseline failed:\n{result.stderr}'

    lines = result.stdout.splitlines()
    mags  = [float(line) for line in lines[1:] if line.strip()]
    return np.array(mags)


def reference_magnitudes(samples):
    # compute fft magnitudes using numpy as the reference
    return np.abs(np.fft.fft(samples))


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