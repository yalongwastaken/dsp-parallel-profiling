#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=results/download_%j.out
#SBATCH --error=results/download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=courses

module load python/3.13.5
source .venv/bin/activate

# soundfile is required for loading flac files
pip install soundfile --quiet

python3 scripts/download_dataset.py