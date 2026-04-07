#!/bin/bash
#SBATCH --job-name=fft_fir_baseline
#SBATCH --output=results/baseline_%j.out
#SBATCH --error=results/baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=courses

# --- configuration -----------------------------------------------------------
RUNS=5                          # number of timed runs per configuration
NUM_TAPS=101                    # fir filter taps
CUTOFF=0.1                      # fir normalized cutoff frequency

# datasets
DATASETS=(
    "data/input_small_downloaded.bin   1048576"    # 2^20
    "data/input_medium_downloaded.bin  16777216"   # 2^24
    "data/input_large_downloaded.bin   67108864"   # 2^26
    "data/input_small_generated.bin   1048576"    # 2^20
    "data/input_medium_generated.bin  16777216"   # 2^24
    "data/input_large_generated.bin   67108864"   # 2^26
)
# -----------------------------------------------------------------------------

module load OpenMPI/4.1.6

# build all targets
cd fft && make && cd ../fir && make && cd ..

echo "=== baseline ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo ""

for entry in "${DATASETS[@]}"; do
    input=$(echo $entry | awk '{print $1}')
    n=$(echo $entry | awk '{print $2}')

    echo "--- dataset: $input (n=$n) ---"

    for run in $(seq 1 $RUNS); do
        ./fft/baseline/fft_baseline $input $n
    done

    for run in $(seq 1 $RUNS); do
        ./fir/baseline/fir_baseline $input $n $NUM_TAPS $CUTOFF
    done

    echo ""
done