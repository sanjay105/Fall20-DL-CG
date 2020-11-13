#!/bin/sh
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time = 04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus =2
#SBATCH --job-name=train_sbanda
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbanda@ufl.edu
#SBATCH --output=serial_%j.out

pwd;hostname;date
module load python
module load torch
module load torchvision
echo "Running script"
python /home/sbanda/
date