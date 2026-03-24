#!/bin/bash
#SBATCH --job-name=fft_fir_pthreads
#SBATCH --output=results/pthreads_%j.out
#SBATCH --error=results/pthreads_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --reservation=courses

# --- configuration -----------------------------------------------------------
RUNS=5                          # number of timed runs per configuration
NUM_TAPS=101                    # fir filter taps
CUTOFF=0.1                      # fir normalized cutoff frequency
THREAD_COUNTS=(1 2 4 8 16 32)  # thread counts to sweep

# dataset sizes — adjust or comment out as needed
DATASETS=(
    "data/input_small.bin   1048576"    # 2^20
    "data/input_medium.bin  16777216"   # 2^24
    "data/input_large.bin   67108864"   # 2^26
)
# -----------------------------------------------------------------------------

module load OpenMPI/4.1.6

echo "=== pthreads ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo ""

for entry in "${DATASETS[@]}"; do
    input=$(echo $entry | awk '{print $1}')
    n=$(echo $entry | awk '{print $2}')

    echo "--- dataset: $input (n=$n) ---"

    for nt in "${THREAD_COUNTS[@]}"; do
        for run in $(seq 1 $RUNS); do
            ./fft/pthreads/fft_pthreads $input $n $nt
        done
    done

    for nt in "${THREAD_COUNTS[@]}"; do
        for run in $(seq 1 $RUNS); do
            ./fir/pthreads/fir_pthreads $input $n $NUM_TAPS $CUTOFF $nt
        done
    done

    echo ""
done