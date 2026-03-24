#!/bin/bash
#SBATCH --job-name=generate_input
#SBATCH --output=results/generate_%j.out
#SBATCH --error=results/generate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=courses

module load python/3.13.5
source .venv/bin/activate

python3 scripts/generate_input.py