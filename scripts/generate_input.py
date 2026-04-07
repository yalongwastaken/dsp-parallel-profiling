# file:        generate_input.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: Generates synthetic audio signals as binary float32 files for FFT/FIR benchmarks.
#              Output files are written to the data/ directory.
# usage:       python3 generate_input.py

import numpy as np
import os

def main():
    os.makedirs("data", exist_ok=True)

    datasets = {
        "small":  2**20,   # ~1M samples
        "medium": 2**24,   # ~16M samples
        "large":  2**26,   # ~67M samples
    }

    sample_rate = 16000  # Hz

    for name, n_samples in datasets.items():
        # generate
        print(f"Generating {name} ({n_samples:,} samples)...", flush=True)
        signal = generate_signal(n_samples, sample_rate)

        # save
        path = f"data/input_{name}_generated.bin"
        signal.tofile(path)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  -> {path} ({size_mb:.1f} MB)")

    print("Done.")

def generate_signal(n_samples: int, sample_rate: int) -> np.ndarray:
    """Generates a mix of sine waves with light noise."""
    t = np.linspace(0, n_samples / sample_rate, n_samples, dtype=np.float32)

    # signal configuratoin
    freqs = [440, 880, 1200, 2500, 5000]
    amps  = [1.0, 0.6, 0.4,  0.3,  0.2]

    signal = np.zeros(n_samples, dtype=np.float32)
    for freq, amp in zip(freqs, amps):
        signal += amp * np.sin(2 * np.pi * freq * t)

    # add some noise
    signal += 0.05 * np.random.default_rng(42).standard_normal(n_samples).astype(np.float32)

    return signal

if __name__ == "__main__":
    main()