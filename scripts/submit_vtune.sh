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
INPUT="data/input_26_generated.bin"   # representative input (2^26)
N=67108864                             # number of samples
NUM_TAPS=101                           # fir taps
CUTOFF=0.1                             # fir cutoff
THREADS=(1 8 32)                       # thread counts to profile
VTUNE_DIR="results/vtune"             # output directory for vtune results
# -----------------------------------------------------------------------------

module load VTune
module load OpenMPI/4.1.6

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
          -collect-with runsa \
          -knob enable-stack-collection=true \
          -result-dir ${VTUNE_DIR}/${label} \
          -- ./fft/pthreads/fft_pthreads $INPUT $N $nt
    vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
          > ${VTUNE_DIR}/${label}_summary.txt
    echo ""
done

# --- fir pthreads ------------------------------------------------------------
for nt in "${THREADS[@]}"; do
    label="fir_pthreads_t${nt}"
    echo "--- $label ---"
    vtune -collect hotspots \
          -collect-with runsa \
          -knob enable-stack-collection=true \
          -result-dir ${VTUNE_DIR}/${label} \
          -- ./fir/pthreads/fir_pthreads $INPUT $N $NUM_TAPS $CUTOFF $nt
    vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
          > ${VTUNE_DIR}/${label}_summary.txt
    echo ""
done

# --- fft openmp --------------------------------------------------------------
label="fft_openmp_t32"
echo "--- $label ---"
vtune -collect hotspots \
      -collect-with runsa \
      -knob enable-stack-collection=true \
      -result-dir ${VTUNE_DIR}/${label} \
      -- ./fft/openmp/fft_openmp $INPUT $N 32
vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
      > ${VTUNE_DIR}/${label}_summary.txt
echo ""

# --- fir openmp --------------------------------------------------------------
label="fir_openmp_t32"
echo "--- $label ---"
vtune -collect hotspots \
      -collect-with runsa \
      -knob enable-stack-collection=true \
      -result-dir ${VTUNE_DIR}/${label} \
      -- ./fir/openmp/fir_openmp $INPUT $N $NUM_TAPS $CUTOFF 32
vtune -report summary -result-dir ${VTUNE_DIR}/${label} \
      > ${VTUNE_DIR}/${label}_summary.txt
echo ""

echo "=== done ==="
echo "summaries written to ${VTUNE_DIR}/*_summary.txt"