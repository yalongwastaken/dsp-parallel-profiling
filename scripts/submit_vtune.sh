#!/bin/bash
#SBATCH --job-name=vtune_profile
#SBATCH --output=results/vtune_%j.out
#SBATCH --error=results/vtune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=courses

# --- configuration -----------------------------------------------------------
REPO="/home/yalong.a/2025-2026/spring/eece5640/finalproject/eece5640-finalproject"
INPUT="${REPO}/data/input_26_generated.bin"   # absolute path to input
N=67108864                                     # number of samples
NUM_TAPS=101                                   # fir taps
CUTOFF=0.1                                     # fir cutoff
THREADS=(1 8 32)                               # thread counts to profile
VTUNE_DIR="${REPO}/results/vtune"             # absolute path for vtune results
# -----------------------------------------------------------------------------

module load VTune/2025.0
module load OpenMPI/4.1.6

# hardcoded repo root — update if your path differs
REPO="/home/yalong.a/2025-2026/spring/eece5640/finalproject/eece5640-finalproject"
cd $REPO

# build all targets
cd fft && make && cd ../fir && make && cd ..

mkdir -p $VTUNE_DIR

echo "=== vtune profiling ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo ""

# --- fft pthreads ------------------------------------------------------------
for nt in "${THREADS[@]}"; do
    label="fft_pthreads_t${nt}"
    echo "--- $label ---"
    vtune -collect hotspots \
          -knob enable-stack-collection=true \
          -result-dir ${VTUNE_DIR}/${label} \
          -- ${REPO}/fft/pthreads/fft_pthreads $INPUT $N $nt
    vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
          > ${VTUNE_DIR}/${label}_summary.txt
    echo ""
done

# --- fir pthreads ------------------------------------------------------------
for nt in "${THREADS[@]}"; do
    label="fir_pthreads_t${nt}"
    echo "--- $label ---"
    vtune -collect hotspots \
          -knob enable-stack-collection=true \
          -result-dir ${VTUNE_DIR}/${label} \
          -- ${REPO}/fir/pthreads/fir_pthreads $INPUT $N $NUM_TAPS $CUTOFF $nt
    vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
          > ${VTUNE_DIR}/${label}_summary.txt
    echo ""
done

# --- fft openmp --------------------------------------------------------------
label="fft_openmp_t32"
echo "--- $label ---"
vtune -collect hotspots \
      -knob enable-stack-collection=true \
      -result-dir ${VTUNE_DIR}/${label} \
      -- ${REPO}/fft/openmp/fft_openmp $INPUT $N 32
vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
      > ${VTUNE_DIR}/${label}_summary.txt
echo ""

# --- fir openmp --------------------------------------------------------------
label="fir_openmp_t32"
echo "--- $label ---"
vtune -collect hotspots \
      -knob enable-stack-collection=true \
      -result-dir ${VTUNE_DIR}/${label} \
      -- ${REPO}/fir/openmp/fir_openmp $INPUT $N $NUM_TAPS $CUTOFF 32
vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
      > ${VTUNE_DIR}/${label}_summary.txt
echo ""

echo "=== done ==="
echo "summaries written to ${VTUNE_DIR}/*_summary.txt"