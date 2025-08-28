#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 2:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --array=0,97

module load conda/latest
conda activate eb_music
python hssm_modelfits.py bootstrap $SLURM_ARRAY_TASK_ID unstructured