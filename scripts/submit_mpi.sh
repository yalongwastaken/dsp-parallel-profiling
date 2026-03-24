#!/bin/bash
#SBATCH --job-name=fft_fir_mpi
#SBATCH --output=results/mpi_%j.out
#SBATCH --error=results/mpi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=courses

# --- configuration -----------------------------------------------------------
RUNS=5                          # number of timed runs per configuration
NUM_TAPS=101                    # fir filter taps
CUTOFF=0.1                      # fir normalized cutoff frequency
PROC_COUNTS=(1 2 4 8 16 32)    # process counts to sweep

# dataset sizes — adjust or comment out as needed
DATASETS=(
    "data/input_small.bin   1048576"    # 2^20
    "data/input_medium.bin  16777216"   # 2^24
    "data/input_large.bin   67108864"   # 2^26
)
# -----------------------------------------------------------------------------

module load OpenMPI/4.1.6

echo "=== mpi ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo ""

for entry in "${DATASETS[@]}"; do
    input=$(echo $entry | awk '{print $1}')
    n=$(echo $entry | awk '{print $2}')

    echo "--- dataset: $input (n=$n) ---"

    for np in "${PROC_COUNTS[@]}"; do
        for run in $(seq 1 $RUNS); do
            mpirun -np $np ./fft/mpi/fft_mpi $input $n
        done
    done

    for np in "${PROC_COUNTS[@]}"; do
        for run in $(seq 1 $RUNS); do
            mpirun -np $np ./fir/mpi/fir_mpi $input $n $NUM_TAPS $CUTOFF
        done
    done

    echo ""
done