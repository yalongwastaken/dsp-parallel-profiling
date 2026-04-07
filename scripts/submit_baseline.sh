#!/bin/bash
#SBATCH --job-name=fft_fir_baseline
#SBATCH --output=results/baseline_%j.out
#SBATCH --error=results/baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=courses

# --- configuration -----------------------------------------------------------
RUNS=5                          # number of timed runs per configuration
NUM_TAPS=101                    # fir filter taps
CUTOFF=0.1                      # fir normalized cutoff frequency

# dataset sizes as exponents — override at submit time with:
# SIZES="20 22 24 26 28" sbatch scripts/submit_baseline.sh
SIZES=${SIZES:-"20 24 26"}

# build dataset array dynamically from exponents
DATASETS=()
for exp in $SIZES; do
    n=$((2**exp))
    DATASETS+=("data/input_${n}_generated.bin   ${n}")
    DATASETS+=("data/input_${n}_downloaded.bin  ${n}")
done
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