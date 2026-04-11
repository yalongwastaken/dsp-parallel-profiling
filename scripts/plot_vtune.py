# file:        plot_vtune.py
# author:      Vaidehi Gohil, Anthony Yalong
# description: parses vtune_summary.csv and generates four profiling figures:
#                1. physical core utilization vs thread count (fft and fir)
#                2. cpu/elapsed time ratio vs thread count (parallelism proxy)
#                3. hotspot stacked bar breakdown for fft at n=2^27
#                4. hotspot stacked bar breakdown for fir at n=2^27
#              output pngs are written to --outdir (default: same dir as csv).
# usage:       python3 scripts/plot_vtune.py --csv results/vtune_summary.csv
#              python3 scripts/plot_vtune.py --csv results/vtune_summary.csv --outdir results/analysis

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

THREAD_COUNTS = [1, 2, 4, 8, 16, 32]
HOTSPOT_N = 134217728       # 2^27 — large enough for stable hotspot %
HOTSPOT_N_LABEL = 27

COLORS = {
    'openmp': '#1f77b4',
    'pthreads': '#ff7f0e',
    'ideal': '#888888',
    'top': '#E53935',
    'second': '#FB8C00',
    'other': '#90A4AE',
}
MARKERS = {'openmp': 'o', 'pthreads': 's'}
LABELS  = {'openmp': 'OpenMP', 'pthreads': 'Pthreads'}

# human-readable names for raw vtune symbol strings
FN_NAME_MAP = {
    'fft_parallel._omp_fn.0': 'fft_parallel (OMP)',
    'fir_parallel._omp_fn.0': 'fir_parallel (OMP)',
    '_IO_fread': 'fread',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',    default='vtune_summary.csv')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)

    print(f'loading {args.csv} ...')
    df = load_data(args.csv)
    print(f'  {len(df)} rows loaded')

    print('generating plots ...')
    plot_core_utilization(df, outdir)
    plot_parallelism_proxy(df, outdir)
    plot_hotspots(df, 'fft', outdir)
    plot_hotspots(df, 'fir', outdir)

    print(f'\ndone. outputs in {outdir}/')

# --- data loading ------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df['threads'] = df['threads'].astype(int)
    df['n']       = df['n'].astype(int)
    # derived: cpu/elapsed ratio — proxy for thread-level parallelism
    df['cpu_elapsed_ratio'] = df['cpu_time_s'] / df['elapsed_time_s']
    # remainder after top two hotspots
    df['other_pct'] = (100.0
                       - df['top_hotspot_pct'].fillna(0)
                       - df['second_hotspot_pct'].fillna(0)).clip(lower=0)
    return df

def mean_by_threads(df, workload, variant, col):
    """return per-thread mean of col for a given workload/variant, indexed to THREAD_COUNTS."""
    sub = df[(df['workload'] == workload) & (df['variant'] == variant)]
    return sub.groupby('threads')[col].mean().reindex(THREAD_COUNTS)

def clean_fn_name(name):
    """map raw vtune symbol names to readable labels."""
    if pd.isna(name) or name == '':
        return 'other'
    return FN_NAME_MAP.get(str(name), str(name))

# --- figure 1: physical core utilization -------------------------------------
def plot_core_utilization(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Physical Core Utilization vs Thread Count', fontsize=13)

    for ax, workload in zip(axes, ['fft', 'fir']):
        for variant in ['openmp', 'pthreads']:
            y = mean_by_threads(df, workload, variant, 'physical_core_util_pct')
            ax.plot(THREAD_COUNTS, y.values,
                    color=COLORS[variant], marker=MARKERS[variant],
                    linewidth=2, markersize=6, label=LABELS[variant])
        ax.set_title(workload.upper(), fontsize=11)
        ax.set_xlabel('thread count')
        ax.set_ylabel('physical cores utilized (%)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(THREAD_COUNTS)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    fig.tight_layout()
    save(fig, outdir, 'vtune_core_utilization.png')

# --- figure 2: cpu/elapsed ratio (parallelism proxy) ------------------------
def plot_parallelism_proxy(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('CPU Time / Elapsed Time vs Thread Count  (ideal = thread count)',
                 fontsize=13)

    for ax, workload in zip(axes, ['fft', 'fir']):
        ax.plot(THREAD_COUNTS, THREAD_COUNTS,
                color=COLORS['ideal'], linestyle='--', linewidth=1.5, label='ideal')
        for variant in ['openmp', 'pthreads']:
            y = mean_by_threads(df, workload, variant, 'cpu_elapsed_ratio')
            ax.plot(THREAD_COUNTS, y.values,
                    color=COLORS[variant], marker=MARKERS[variant],
                    linewidth=2, markersize=6, label=LABELS[variant])
        ax.set_title(workload.upper(), fontsize=11)
        ax.set_xlabel('thread count')
        ax.set_ylabel('cpu time / elapsed time')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xticks(THREAD_COUNTS)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    fig.tight_layout()
    save(fig, outdir, 'vtune_parallelism_proxy.png')

# --- figures 3 & 4: hotspot stacked bars ------------------------------------
def plot_hotspots(df, workload, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        f'{workload.upper()} Hotspot Breakdown  n=2^{HOTSPOT_N_LABEL} ({HOTSPOT_N:,})',
        fontsize=13)

    for ax, variant in zip(axes, ['openmp', 'pthreads']):
        sub = (df[(df['workload'] == workload) &
                  (df['variant']  == variant)  &
                  (df['n']        == HOTSPOT_N)]
               .sort_values('threads'))

        if sub.empty:
            ax.set_title(f'{LABELS[variant]} (no data)')
            continue

        threads = sub['threads'].tolist()
        top_pct = sub['top_hotspot_pct'].fillna(0).tolist()
        second_pct = sub['second_hotspot_pct'].fillna(0).tolist()
        other_pct = sub['other_pct'].tolist()
        bottoms = [a + b for a, b in zip(top_pct, second_pct)]

        # use modal function name across thread counts as legend label
        top_label = clean_fn_name(sub['top_hotspot_fn'].mode().iloc[0] if not sub['top_hotspot_fn'].isna().all() else '')
        second_label = clean_fn_name(sub['second_hotspot_fn'].mode().iloc[0] if not sub['second_hotspot_fn'].isna().all() else '')

        x  = np.arange(len(threads))
        width = 0.55

        ax.bar(x, top_pct, width, label=top_label, color=COLORS['top'])
        ax.bar(x, second_pct, width, bottom=top_pct, label=second_label, color=COLORS['second'])
        ax.bar(x, other_pct, width, bottom=bottoms, label='other', color=COLORS['other'])

        ax.set_title(LABELS[variant], fontsize=11)
        ax.set_xlabel('thread count')
        ax.set_ylabel('% of cpu time')
        ax.set_xticks(x)
        ax.set_xticklabels(threads)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    save(fig, outdir, f'vtune_hotspots_{workload}.png')

# --- helpers -----------------------------------------------------------------
def save(fig, outdir, fname):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


if __name__ == '__main__':
    main()