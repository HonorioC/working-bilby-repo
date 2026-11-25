#!/bin/bash 
#SBATCH --job-name=Bilby_Job
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=48:00:00
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --nodes=1


module purge
module load anaconda3/2026.7
conda activate bilby-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python GW150914.py #script ur running
