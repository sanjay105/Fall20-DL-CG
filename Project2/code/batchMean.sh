#!/bin/sh
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=train_mean_sbanda
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbanda@ufl.edu
#SBATCH --output=serial_%j.out

pwd;hostname;date
module load python
module load torch
module load torchvision
echo "Running script"
python /home/sbanda/Fall20-DL-CG/Project2/Finale/main_mean.py
date