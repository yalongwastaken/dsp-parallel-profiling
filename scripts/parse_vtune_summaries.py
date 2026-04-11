# parse_vtune_summaries.py
# parses all vtune summary txt files into a single csv
# usage: python parse_vtune_summaries.py <vtune_dir> <output_csv>

import re
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <vtune_dir> <output_csv>")
        sys.exit(1)

    vtune_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    summary_files = sorted(vtune_dir.glob("*_summary.txt"))
    if not summary_files:
        print(f"no summary files found in {vtune_dir}")
        sys.exit(1)

    rows = []
    for path in summary_files:
        row = parse_summary(path)
        if row:
            rows.append(row)

    if not rows:
        print("no data parsed")
        sys.exit(1)

    fieldnames = [
        "workload", "variant", "threads", "n",
        "elapsed_time_s", "cpu_time_s",
        "top_hotspot_fn", "top_hotspot_pct",
        "second_hotspot_fn", "second_hotspot_pct",
        "physical_core_util_pct", "filename",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_csv}")


def parse_summary(path):
    # filename format: fft_pthreads_t8_n67108864_summary.txt
    name = path.stem.replace("_summary", "")
    match = re.match(r"(fft|fir)_(pthreads|openmp)_t(\d+)_n(\d+)", name)
    if not match:
        print(f"skipping unrecognized filename: {path.name}")
        return None

    workload  = match.group(1)   # fft | fir
    variant   = match.group(2)   # pthreads | openmp
    threads   = int(match.group(3))
    n         = int(match.group(4))

    text = path.read_text(errors="replace")

    elapsed   = extract_float(r"Elapsed Time:\s+([\d.]+)s", text)
    cpu_time  = extract_float(r"CPU Time:\s+([\d.]+)s", text)
    phys_util = extract_float(r"Effective Physical Core Utilization:\s+([\d.]+)%", text)

    # top hotspots table — grab first two function rows
    hotspot_rows = re.findall(
        r"^(\S+)\s+\S+\s+([\d.]+)s\s+([\d.]+)%",
        text, re.MULTILINE
    )

    top_fn   = hotspot_rows[0][0] if len(hotspot_rows) > 0 else ""
    top_pct  = float(hotspot_rows[0][2]) if len(hotspot_rows) > 0 else None
    sec_fn   = hotspot_rows[1][0] if len(hotspot_rows) > 1 else ""
    sec_pct  = float(hotspot_rows[1][2]) if len(hotspot_rows) > 1 else None

    return {
        "workload":            workload,
        "variant":             variant,
        "threads":             threads,
        "n":                   n,
        "elapsed_time_s":      elapsed,
        "cpu_time_s":          cpu_time,
        "top_hotspot_fn":      top_fn,
        "top_hotspot_pct":     top_pct,
        "second_hotspot_fn":   sec_fn,
        "second_hotspot_pct":  sec_pct,
        "physical_core_util_pct": phys_util,
        "filename":            path.name,
    }


def extract_float(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    main()