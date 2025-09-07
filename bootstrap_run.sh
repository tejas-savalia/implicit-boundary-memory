#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 2:00:00  # Job time limit
#SBATCH -o slurm_logs/slurm-%j.out  # %j = job ID
#SBATCH --array=26

module load conda/latest
conda activate eb_music
python indiv_fits.py $SLURM_ARRAY_TASK_ID