#!/bin/bash
#SBATCH --job-name=vtune_profile
#SBATCH --output=results/vtune_%j.out
#SBATCH --error=results/vtune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=courses

# --- configuration -----------------------------------------------------------
REPO="/home/yalong.a/2025-2026/spring/eece5640/finalproject/eece5640-finalproject"
NUM_TAPS=101                                   # fir taps
CUTOFF=0.1                                     # fir cutoff
THREADS=(1 2 4 8 16 32)                        # thread counts to profile
VTUNE_DIR="${REPO}/results/vtune"              # absolute path for vtune results

# dataset sizes as exponents — override at submit time with:
# SIZES="20 22 24 26" sbatch scripts/submit_vtune.sh
SIZES=${SIZES:-"26"}

# build dataset array dynamically from exponents
DATASETS=()
for exp in $SIZES; do
    n=$((2**exp))
    DATASETS+=("data/input_${exp}_generated.bin ${n}")
    DATASETS+=("data/input_${exp}_downloaded.bin ${n}")
done
# -----------------------------------------------------------------------------

module load VTune/2025.0
module load OpenMPI/4.1.6

cd $REPO
cd fft && make && cd ../fir && make && cd ..

mkdir -p $VTUNE_DIR

echo "=== vtune profiling ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo ""

for entry in "${DATASETS[@]}"; do
    input=$(echo $entry | awk '{print $1}')
    n=$(echo $entry | awk '{print $2}')
    INPUT="${REPO}/${input}"

    if [[ ! -f "$INPUT" ]]; then
        echo "skipping missing input: $INPUT"
        echo ""
        continue
    fi

    echo "=== dataset: $input (n=$n) ==="
    echo ""

    # --- fft pthreads --------------------------------------------------------
    for nt in "${THREADS[@]}"; do
        label="fft_pthreads_t${nt}_n${n}"
        echo "--- $label ---"
        vtune -collect hotspots \
              -knob enable-stack-collection=true \
              -result-dir ${VTUNE_DIR}/${label} \
              -- ${REPO}/fft/pthreads/fft_pthreads $INPUT $n $nt
        vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
              > ${VTUNE_DIR}/${label}_summary.txt
        echo ""
    done

    # --- fir pthreads --------------------------------------------------------
    for nt in "${THREADS[@]}"; do
        label="fir_pthreads_t${nt}_n${n}"
        echo "--- $label ---"
        vtune -collect hotspots \
              -knob enable-stack-collection=true \
              -result-dir ${VTUNE_DIR}/${label} \
              -- ${REPO}/fir/pthreads/fir_pthreads $INPUT $n $NUM_TAPS $CUTOFF $nt
        vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
              > ${VTUNE_DIR}/${label}_summary.txt
        echo ""
    done

    # --- fft openmp ----------------------------------------------------------
    for nt in "${THREADS[@]}"; do
        label="fft_openmp_t${nt}_n${n}"
        echo "--- $label ---"
        vtune -collect hotspots \
              -knob enable-stack-collection=true \
              -result-dir ${VTUNE_DIR}/${label} \
              -- ${REPO}/fft/openmp/fft_openmp $INPUT $n $nt
        vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
              > ${VTUNE_DIR}/${label}_summary.txt
        echo ""
    done

    # --- fir openmp ----------------------------------------------------------
    for nt in "${THREADS[@]}"; do
        label="fir_openmp_t${nt}_n${n}"
        echo "--- $label ---"
        vtune -collect hotspots \
              -knob enable-stack-collection=true \
              -result-dir ${VTUNE_DIR}/${label} \
              -- ${REPO}/fir/openmp/fir_openmp $INPUT $n $NUM_TAPS $CUTOFF $nt
        vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
              > ${VTUNE_DIR}/${label}_summary.txt
        echo ""
    done

done

echo "=== done ==="
echo "summaries written to ${VTUNE_DIR}/*_summary.txt"