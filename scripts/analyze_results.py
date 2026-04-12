# file:        analyze_results.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: parses raw benchmark output files, computes mean/std dev,
#              speedup, and parallel efficiency, writes a combined CSV,
#              and generates speedup, efficiency, scaling, and weak scaling plots.
# usage:       python3 scripts/analyze_results.py \
#                  --baseline  results/baseline.out \
#                  --pthreads  results/pthreads.out \
#                  --openmp    results/openmp.out \
#                  --mpi       results/mpi.out \
#                  --outdir    results/analysis

import re
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# --- parsing -----------------------------------------------------------------

def parse_file(path):
    """
    Parse a benchmark output file into a dict keyed by (workload, middleware,
    input_type, n, p) where p is thread/process count (1 for baseline).

    Returns dict mapping key -> list of float timing values (ms).
    """
    data = defaultdict(list)
    if not path or not os.path.exists(path):
        return data

    # patterns for each middleware
    patterns = [
        # fft_baseline n=1048576 time_ms=110.05
        (r'(fft|fir)_baseline\s+n=(\d+)(?:\s+taps=\d+)?\s+time_ms=([\d.]+)',
         lambda m: ('fft' if m.group(1) == 'fft' else 'fir', 'baseline', 1)),
        # fft_pthreads n=1048576 threads=4 time_ms=75.2
        (r'(fft|fir)_pthreads\s+n=(\d+)(?:\s+taps=\d+)?\s+threads=(\d+)\s+time_ms=([\d.]+)',
         lambda m: (m.group(1), 'pthreads', int(m.group(3)))),
        # fft_openmp n=1048576 threads=4 time_ms=75.2
        (r'(fft|fir)_openmp\s+n=(\d+)(?:\s+taps=\d+)?\s+threads=(\d+)\s+time_ms=([\d.]+)',
         lambda m: (m.group(1), 'openmp', int(m.group(3)))),
        # fft_mpi n=1048576 nprocs=4 chunk=... time_ms=18.9
        (r'(fft|fir)_mpi\s+n=(\d+)(?:\s+taps=\d+)?\s+nprocs=(\d+)(?:\s+chunk=\d+)?\s+time_ms=([\d.]+)',
         lambda m: (m.group(1), 'mpi', int(m.group(3)))),
    ]

    # track which dataset (input type) we are currently in
    current_input_type = 'generated'  # default

    with open(path) as f:
        for line in f:
            line = line.strip()

            # detect dataset header to determine input type
            if line.startswith('--- dataset:'):
                if 'downloaded' in line:
                    current_input_type = 'downloaded'
                else:
                    current_input_type = 'generated'
                continue

            for pattern, key_fn in patterns:
                m = re.match(pattern, line)
                if m:
                    workload, middleware, p = key_fn(m)
                    n       = int(m.group(2))
                    # time_ms is always the last capture group
                    time_ms = float(m.groups()[-1])
                    key = (workload, middleware, current_input_type, n, p)
                    data[key].append(time_ms)
                    break

    return data


def merge(*dicts):
    """merge multiple parsed data dicts into one."""
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].extend(v)
    return merged


# --- statistics --------------------------------------------------------------

def compute_stats(data):
    """
    Compute mean, std dev, speedup, and efficiency for all configurations.

    Speedup S(p) = T(1) / T(p) where T(1) is the baseline mean for that
    workload, input_type, and n. For MPI, T(1) is mpi nprocs=1 (segmented
    FFT at full n), not the serial baseline, since the compute differs.

    Returns list of dicts, one per configuration.
    """
    rows = []

    # group by (workload, input_type, n) to find T(1) references
    groups = defaultdict(dict)
    for (workload, middleware, input_type, n, p), times in data.items():
        if not times:
            continue
        mean = np.mean(times)
        std  = np.std(times, ddof=1) if len(times) > 1 else 0.0
        groups[(workload, input_type, n)][(middleware, p)] = (mean, std)

    for (workload, input_type, n), configs in groups.items():
        # serial reference: baseline p=1, or pthreads/openmp p=1 if no baseline
        t1_serial = None
        if ('baseline', 1) in configs:
            t1_serial = configs[('baseline', 1)][0]

        for (middleware, p), (mean, std) in sorted(configs.items(),
                                                    key=lambda x: x[0]):
            # choose T(1) for speedup computation
            if middleware == 'mpi':
                # MPI speedup relative to mpi nprocs=1 (same segmented FFT)
                t1 = configs.get(('mpi', 1), (None, None))[0]
            else:
                # pthreads/openmp speedup relative to serial baseline
                t1 = t1_serial

            speedup    = (t1 / mean) if (t1 and mean) else None
            efficiency = (speedup / p) if (speedup and p > 0) else None

            rows.append({
                'workload':    workload,
                'middleware':  middleware,
                'input_type':  input_type,
                'n':           n,
                'p':           p,
                'mean_ms':     round(mean, 4),
                'std_ms':      round(std, 4),
                'speedup':     round(speedup, 4) if speedup else '',
                'efficiency':  round(efficiency, 4) if efficiency else '',
            })

    return rows


# --- csv output --------------------------------------------------------------

def write_csv(rows, path):
    """write stats rows to a CSV file."""
    fields = ['workload', 'middleware', 'input_type', 'n', 'p',
              'mean_ms', 'std_ms', 'speedup', 'efficiency']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


# --- plotting ----------------------------------------------------------------

MIDDLEWARES = ['pthreads', 'openmp', 'mpi']
COLORS      = {'pthreads': '#1f77b4', 'openmp': '#ff7f0e', 'mpi': '#2ca02c'}
MARKERS     = {'pthreads': 'o', 'openmp': 's', 'mpi': '^'}

# representative sizes to show on plots — skip very small n where noise dominates
PLOT_SIZES  = [2**e for e in [22, 23, 24, 25, 26, 27, 28, 29]]

# weak scaling: base size n0 at p=1, scaled as n0*p for each p
# 2^23 * 32 = 2^28, which is within the data range for all middlewares
WEAK_BASE_EXP = 23
WEAK_PAIRS    = [(p, 2**WEAK_BASE_EXP * p) for p in [1, 2, 4, 8, 16, 32]]


def plot_speedup(rows, workload, input_type, outdir):
    """plot speedup vs thread/process count for selected n values."""
    fig, axes = plt.subplots(1, len(PLOT_SIZES), figsize=(4 * len(PLOT_SIZES), 4),
                             sharey=False)
    fig.suptitle(f'{workload.upper()} speedup — {input_type} input', fontsize=13)

    for ax, n in zip(axes, PLOT_SIZES):
        ax.set_title(f'n = 2^{int(np.log2(n))}', fontsize=10)
        ax.set_xlabel('threads / processes')
        ax.set_ylabel('speedup S(p)')

        # ideal speedup reference line
        ps = [1, 2, 4, 8, 16, 32]
        ax.plot(ps, ps, 'k--', linewidth=0.8, label='ideal', alpha=0.5)

        for mw in MIDDLEWARES:
            pts = [(r['p'], r['speedup'])
                   for r in rows
                   if r['workload'] == workload
                   and r['middleware'] == mw
                   and r['input_type'] == input_type
                   and r['n'] == n
                   and r['speedup'] != ''
                   and r['p'] > 0]
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=MARKERS[mw], color=COLORS[mw],
                    label=mw, linewidth=1.5, markersize=5)

        ax.set_xticks(ps)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(outdir, f'speedup_{workload}_{input_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


def plot_efficiency(rows, workload, input_type, outdir):
    """plot parallel efficiency vs thread/process count for selected n values."""
    fig, axes = plt.subplots(1, len(PLOT_SIZES), figsize=(4 * len(PLOT_SIZES), 4),
                             sharey=False)
    fig.suptitle(f'{workload.upper()} parallel efficiency — {input_type} input',
                 fontsize=13)

    for ax, n in zip(axes, PLOT_SIZES):
        ax.set_title(f'n = 2^{int(np.log2(n))}', fontsize=10)
        ax.set_xlabel('threads / processes')
        ax.set_ylabel('efficiency E(p) = S(p) / p')
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8,
                   label='ideal', alpha=0.5)

        for mw in MIDDLEWARES:
            pts = [(r['p'], r['efficiency'])
                   for r in rows
                   if r['workload'] == workload
                   and r['middleware'] == mw
                   and r['input_type'] == input_type
                   and r['n'] == n
                   and r['efficiency'] != ''
                   and r['p'] > 0]
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=MARKERS[mw], color=COLORS[mw],
                    label=mw, linewidth=1.5, markersize=5)

        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(outdir, f'efficiency_{workload}_{input_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


def plot_scaling(rows, workload, input_type, outdir):
    """plot mean execution time vs n for each middleware at 32 threads."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f'{workload.upper()} strong scaling (p=32) — {input_type} input',
                 fontsize=12)
    ax.set_xlabel('n (log2 scale)')
    ax.set_ylabel('mean time (ms)')

    for mw in ['baseline'] + MIDDLEWARES:
        p = 1 if mw == 'baseline' else 32
        pts = [(r['n'], r['mean_ms'])
               for r in rows
               if r['workload'] == workload
               and r['middleware'] == mw
               and r['input_type'] == input_type
               and r['p'] == p
               and r['mean_ms'] > 0]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        color = '#888888' if mw == 'baseline' else COLORS[mw]
        marker = 'x' if mw == 'baseline' else MARKERS[mw]
        ax.plot(xs, ys, marker=marker, color=color, label=mw,
                linewidth=1.5, markersize=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'2^{int(np.log2(x))}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(outdir, f'scaling_{workload}_{input_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


def plot_generated_vs_downloaded(rows, workload, middleware, outdir):
    """plot mean time for generated vs downloaded input at p=32 across n."""
    p = 1 if middleware == 'baseline' else 32
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f'{workload.upper()} {middleware} — generated vs downloaded (p={p})',
                 fontsize=11)
    ax.set_xlabel('n (log2 scale)')
    ax.set_ylabel('mean time (ms)')

    for itype, color, ls in [('generated', '#1f77b4', '-'),
                              ('downloaded', '#d62728', '--')]:
        pts = [(r['n'], r['mean_ms'])
               for r in rows
               if r['workload'] == workload
               and r['middleware'] == middleware
               and r['input_type'] == itype
               and r['p'] == p
               and r['mean_ms'] > 0]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker='o', color=color, linestyle=ls,
                label=itype, linewidth=1.5, markersize=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'2^{int(np.log2(x))}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(outdir, f'genvsdown_{workload}_{middleware}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


def plot_weak_scaling(rows, workload, input_type, outdir):
    """plot weak scaling efficiency vs process count.

    weak scaling holds work-per-process constant by pairing each p with
    n = WEAK_BASE * p. efficiency = T(1, n0) / T(p, n0*p) — ideal is 1.0.
    uses generated input only since all sizes are present there.
    """
    # build a lookup: (middleware, n, p) -> mean_ms
    lookup = {(r['middleware'], r['n'], r['p']): r['mean_ms']
              for r in rows
              if r['workload'] == workload and r['input_type'] == input_type}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'{workload.upper()} weak scaling efficiency — {input_type} input'
        f'  (base n = 2^{WEAK_BASE_EXP} per process)',
        fontsize=13)

    ps = [p for p, _ in WEAK_PAIRS]

    for ax, panel in zip(axes, ['efficiency', 'time']):
        ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8,
                   label='ideal', alpha=0.5)

        for mw in MIDDLEWARES:
            # T(1) at base size for this middleware
            t1 = lookup.get((mw, WEAK_PAIRS[0][1], 1))
            if t1 is None:
                continue

            pts = []
            for p, n in WEAK_PAIRS:
                t = lookup.get((mw, n, p))
                if t is None:
                    continue
                if panel == 'efficiency':
                    pts.append((p, t1 / t))   # weak efficiency = T(1) / T(p)
                else:
                    pts.append((p, t))         # raw time to show overhead growth

            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=MARKERS[mw], color=COLORS[mw],
                    label=mw, linewidth=1.5, markersize=5)

        ax.set_xlabel('process / thread count')
        ax.set_xscale('log', base=2)
        ax.set_xticks(ps)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend()
        ax.grid(True, alpha=0.3)

        if panel == 'efficiency':
            ax.set_title('weak scaling efficiency  E = T(1) / T(p)', fontsize=10)
            ax.set_ylabel('efficiency (1.0 = ideal)')
            ax.set_ylim(0, 1.2)
        else:
            ax.set_title('wall time at each (p, n=base*p)', fontsize=10)
            ax.set_ylabel('mean time (ms)')

    # annotate the (p, n) pairs along the x-axis of the time panel
    axes[1].set_xticks(ps)
    axes[1].set_xticklabels(
        [f'p={p}\nn=2^{int(np.log2(n))}' for p, n in WEAK_PAIRS],
        fontsize=7)

    plt.tight_layout()
    fname = os.path.join(outdir, f'weak_scaling_{workload}_{input_type}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',  default=None)
    ap.add_argument('--pthreads',  default=None)
    ap.add_argument('--openmp',    default=None)
    ap.add_argument('--mpi',       default=None)
    ap.add_argument('--outdir',    default='results/analysis')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("parsing result files ...")
    data = merge(
        parse_file(args.baseline),
        parse_file(args.pthreads),
        parse_file(args.openmp),
        parse_file(args.mpi),
    )
    print(f"  {len(data)} configurations found")

    print("computing statistics ...")
    rows = compute_stats(data)
    print(f"  {len(rows)} rows computed")

    print("writing CSV ...")
    write_csv(rows, os.path.join(args.outdir, 'results.csv'))

    print("generating plots ...")
    for workload in ['fft', 'fir']:
        for input_type in ['generated', 'downloaded']:
            plot_speedup(rows, workload, input_type, args.outdir)
            plot_efficiency(rows, workload, input_type, args.outdir)
            plot_scaling(rows, workload, input_type, args.outdir)
            plot_weak_scaling(rows, workload, input_type, args.outdir)
        for middleware in ['baseline', 'pthreads', 'openmp', 'mpi']:
            plot_generated_vs_downloaded(rows, workload, middleware, args.outdir)

    print(f"\ndone. outputs in {args.outdir}/")


if __name__ == "__main__":
    main()