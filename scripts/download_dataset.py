# file:        download_dataset.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: downloads a librispeech audio file, concatenates enough audio to
#              meet the required sample counts, resamples to 16kHz, and writes
#              binary float32 files matching the specified dataset sizes.
#              dataset sizes are configurable via the SIZES environment variable
#              as a space-separated list of exponents (e.g. SIZES="20 24 26")
# usage:       python3 scripts/download_dataset.py
#              SIZES="20 22 24 26 28" python3 scripts/download_dataset.py

import os
import ssl
import urllib.request
import tarfile
import numpy as np

# target sample counts — configurable via SIZES env var (space-separated exponents)
# default: 20 24 26
SIZES_ENV = os.environ.get("SIZES", "20 24 26")
DATASETS  = {2**int(e): 2**int(e) for e in SIZES_ENV.split()}

TARGET_SR   = 16000             # hz — matches generate_input.py
OUTPUT_DIR  = "data"
TARBALL     = "data/dev-clean.tar.gz"
EXTRACT_DIR = "data/librispeech_tmp"

# librispeech dev-clean subset — small enough to download quickly (~337MB)
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

# bypass ssl verification — required on macOS with python.org installer
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def download_tarball(url, dest):
    """Download tarball from url to dest in chunks if not already present."""
    if os.path.exists(dest):
        print(f"  already downloaded: {dest}")
        return
    print(f"  downloading {url} ...")
    with urllib.request.urlopen(url, context=SSL_CONTEXT) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        with open(dest, "wb") as f:
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.1f}%)", flush=True)
    print(f"  saved to {dest}")


def extract_tarball(tarball, dest):
    """Extract tarball to dest directory if not already extracted."""
    if os.path.exists(dest):
        print(f"  already extracted: {dest}")
        return
    print(f"  extracting {tarball} ...")
    with tarfile.open(tarball, "r:gz") as t:
        t.extractall(dest)
    print(f"  extracted to {dest}")


def load_flac_files(root):
    """
    Walk root directory and load all .flac files into a single float32 array.

    Returns concatenated samples normalized to [-1, 1].
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("pip install soundfile")

    all_samples = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith(".flac"):
                path = os.path.join(dirpath, fname)
                data, sr = sf.read(path, dtype="float32")
                # resample to target sr if needed
                if sr != TARGET_SR:
                    data = resample(data, sr, TARGET_SR)
                # convert stereo to mono by averaging channels
                if data.ndim > 1:
                    data = data.mean(axis=1)
                all_samples.append(data)

    return np.concatenate(all_samples)


def resample(data, orig_sr, target_sr):
    """Resample data from orig_sr to target_sr using linear interpolation."""
    if orig_sr == target_sr:
        return data
    ratio       = target_sr / orig_sr
    n_out       = int(len(data) * ratio)
    x_orig      = np.linspace(0, len(data) - 1, len(data))
    x_resampled = np.linspace(0, len(data) - 1, n_out)
    return np.interp(x_resampled, x_orig, data).astype(np.float32)


def write_binary(samples, path):
    """Write float32 samples to binary file."""
    samples.astype(np.float32).tofile(path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  -> {path} ({size_mb:.1f} MB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # download and extract librispeech dev-clean into data/
    download_tarball(LIBRISPEECH_URL, TARBALL)
    extract_tarball(TARBALL, EXTRACT_DIR)

    # load all audio into one long array
    print("loading audio files ...")
    audio = load_flac_files(EXTRACT_DIR)
    print(f"  total samples loaded: {len(audio):,} ({len(audio)/TARGET_SR:.1f}s at {TARGET_SR}Hz)")

    # write each dataset by tiling audio if not long enough
    for n_samples, _ in DATASETS.items():
        print(f"generating n=2^{int(np.log2(n_samples))} ({n_samples:,} samples) ...")

        if len(audio) < n_samples:
            # tile audio to reach required length
            repeats = int(np.ceil(n_samples / len(audio)))
            tiled   = np.tile(audio, repeats)
        else:
            tiled = audio

        # trim to exact size
        out = tiled[:n_samples].astype(np.float32)
        write_binary(out, os.path.join(OUTPUT_DIR, f"input_{n_samples}_downloaded.bin"))

    print("done.")
    print(f"\nnote: temp files left in {EXTRACT_DIR}/ and {TARBALL}")
    print("      remove them manually to free space once data/ is verified")


if __name__ == "__main__":
    main()