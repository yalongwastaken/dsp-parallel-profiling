# file:        generate_input.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: generates synthetic audio signals as binary float32 files for FFT/FIR benchmarks.
#              output files are written to the data/ directory.
#              dataset sizes are configurable via the SIZES environment variable
#              as a space-separated list of exponents (e.g. SIZES="20 24 26")
# usage:       python3 generate_input.py
#              SIZES="20 22 24 26 28" python3 generate_input.py

import numpy as np
import os


def main():
    os.makedirs("data", exist_ok=True)

    # read sizes from environment variable, default to 20 24 26
    sizes_env = os.environ.get("SIZES", "20 24 26")
    exponents = [int(e) for e in sizes_env.split()]

    datasets = {str(2**e): 2**e for e in exponents}

    sample_rate = 16000     # hz

    for name, n_samples in datasets.items():
        print(f"generating n=2^{int(np.log2(n_samples))} ({n_samples:,} samples)...", flush=True)
        signal = generate_signal(n_samples, sample_rate)

        path = f"data/input_{n_samples}_generated.bin"
        signal.tofile(path)

        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  -> {path} ({size_mb:.1f} MB)")

    print("done.")


def generate_signal(n_samples: int, sample_rate: int) -> np.ndarray:
    """Generate a mix of sine waves at audio frequencies with light noise."""
    t = np.linspace(0, n_samples / sample_rate, n_samples, dtype=np.float32)

    # mix of frequencies spanning low to high audio range
    freqs = [440, 880, 1200, 2500, 5000]
    amps  = [1.0, 0.6,  0.4,  0.3,  0.2]

    signal = np.zeros(n_samples, dtype=np.float32)
    for freq, amp in zip(freqs, amps):
        signal += amp * np.sin(2 * np.pi * freq * t)

    # small amount of gaussian noise to simulate real-world signal
    signal += 0.05 * np.random.default_rng(42).standard_normal(n_samples).astype(np.float32)

    return signal


if __name__ == "__main__":
    main()