# test_fir_basline.py
# author: Vaidehi Gohil, Anthony Yalong
# runs correctness tests for fir_serial against scipy.signal as reference

import subprocess
import struct
import tempfile
import os
import numpy as np

# path to compiled binary relative to this file
FIR_BIN = os.path.join(os.path.dirname(__file__), "../../../fir/baseline/fir_baseline")

# default filter params used across tests
NUM_TAPS    = 101
CUTOFF      = 0.1   # normalized, i.e. fc / fs

# tolerance for comparing output samples against scipy reference
ABS_TOL = 1e-3
REL_TOL = 1e-4


def write_binary(samples):
    """Write float32 samples to a temp binary file, return the file path."""
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    f.write(struct.pack(f"{len(samples)}f", *samples))
    f.close()
    return f.name


def run_fir_serial(bin_path, n, num_taps=NUM_TAPS, cutoff=CUTOFF):
    """
    Run fir_serial with -v and parse output samples from stdout.

    Returns numpy array of float32 output samples.
    """
    result = subprocess.run(
        [FIR_BIN, bin_path, str(n), str(num_taps), str(cutoff), "-v"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"fir_serial failed:\n{result.stderr}"

    # first line is the timing summary; remaining lines are one sample per line
    lines = result.stdout.splitlines()
    out = [float(line) for line in lines[1:] if line.strip()]
    return np.array(out, dtype=np.float32)


def reference_fir(samples, num_taps=NUM_TAPS, cutoff=CUTOFF):
    """
    Compute fir filter output using the same hamming window method as fir_baseline.

    Matches the C implementation exactly: sinc * hamming, normalized by coefficient sum.
    """
    M = num_taps - 1
    h = np.zeros(num_taps)
    for i in range(num_taps):
        hamming = 0.54 - 0.46 * np.cos(2.0 * np.pi * i / M)
        if i == M // 2:
            sinc = 2.0 * cutoff
        else:
            sinc = np.sin(2.0 * np.pi * cutoff * (i - M / 2.0)) / (np.pi * (i - M / 2.0))
        h[i] = hamming * sinc
    h /= h.sum()

    # convolve with zero-padding on the left to match C boundary behavior
    out = np.zeros(len(samples))
    for i in range(len(samples)):
        for k in range(num_taps):
            if i - k >= 0:
                out[i] += h[k] * samples[i - k]
    return out.astype(np.float32)


def check(label, got, expected):
    """Assert got and expected agree within tolerance, print pass/fail."""
    close = np.allclose(got, expected, atol=ABS_TOL, rtol=REL_TOL)
    status = "PASS" if close else "FAIL"
    print(f"  [{status}] {label}")
    if not close:
        worst = np.max(np.abs(got - expected))
        print(f"         max absolute error: {worst:.6e}")
    assert close, f"test failed: {label}"


# --- tests -------------------------------------------------------------------

def test_low_frequency_passed():
    # low-frequency sine below cutoff — should pass through mostly unchanged
    n = 1024
    t = np.arange(n)
    # bin 5 is well below cutoff of 0.1 * fs
    samples = np.sin(2 * np.pi * 5 * t / n).astype(np.float32)

    path = write_binary(samples)
    try:
        got = run_fir_serial(path, n)
        expected = reference_fir(samples)
        check("low-frequency sine passes through (n=1024, bin=5)", got, expected)
    finally:
        os.unlink(path)


def test_high_frequency_attenuated():
    # high-frequency sine above cutoff — output energy should be much lower than input
    n = 1024
    t = np.arange(n)
    # bin 400 is well above cutoff of 0.1 * fs
    samples = np.sin(2 * np.pi * 400 * t / n).astype(np.float32)

    path = write_binary(samples)
    try:
        got = run_fir_serial(path, n)
        expected = reference_fir(samples)
        check("high-frequency sine is attenuated (n=1024, bin=400)", got, expected)

        # also verify attenuation vs raw input energy
        input_rms  = np.sqrt(np.mean(samples[NUM_TAPS:]**2))
        output_rms = np.sqrt(np.mean(got[NUM_TAPS:]**2))
        attenuated = output_rms < input_rms * 0.1
        status = "PASS" if attenuated else "FAIL"
        print(f"  [{status}] high-frequency attenuation check "
              f"(in_rms={input_rms:.4f}, out_rms={output_rms:.4f})")
        assert attenuated, "high-frequency signal was not sufficiently attenuated"
    finally:
        os.unlink(path)


def test_all_zeros():
    # zero input — output should also be zero
    n = 512
    samples = np.zeros(n, dtype=np.float32)

    path = write_binary(samples)
    try:
        got = run_fir_serial(path, n)
        expected = reference_fir(samples)
        check("all zeros (n=512)", got, expected)
    finally:
        os.unlink(path)


def test_impulse_response():
    # unit impulse at index 0 — output is the filter's impulse response
    n = 512
    samples = np.zeros(n, dtype=np.float32)
    samples[0] = 1.0

    path = write_binary(samples)
    try:
        got = run_fir_serial(path, n)
        expected = reference_fir(samples)
        check("impulse response (n=512)", got, expected)
    finally:
        os.unlink(path)


def test_dc_signal():
    # constant dc input — after transient, output should equal input (unity dc gain)
    n = 1024
    samples = np.ones(n, dtype=np.float32) * 2.0

    path = write_binary(samples)
    try:
        got = run_fir_serial(path, n)
        expected = reference_fir(samples)
        check("dc signal steady state (n=1024, value=2.0)", got, expected)

        # verify steady-state output (skip filter transient of num_taps samples)
        steady = got[NUM_TAPS:]
        close_to_input = np.allclose(steady, 2.0, atol=1e-2)
        status = "PASS" if close_to_input else "FAIL"
        print(f"  [{status}] dc unity gain check "
              f"(mean steady-state={np.mean(steady):.6f}, expected=2.0)")
        assert close_to_input, "dc gain is not unity in steady state"
    finally:
        os.unlink(path)


def test_random_signal():
    # random broadband signal — general correctness against scipy reference
    n = 2048
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(n).astype(np.float32)

    path = write_binary(samples)
    try:
        got = run_fir_serial(path, n)
        expected = reference_fir(samples)
        check("random broadband signal (n=2048, seed=42)", got, expected)
    finally:
        os.unlink(path)


# --- entry point -------------------------------------------------------------

def main():
    tests = [
        test_low_frequency_passed,
        test_high_frequency_attenuated,
        test_all_zeros,
        test_impulse_response,
        test_dc_signal,
        test_random_signal,
    ]

    print("=== fir_serial correctness tests ===")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            pass

    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()