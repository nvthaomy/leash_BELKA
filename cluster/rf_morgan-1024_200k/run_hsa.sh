#!/bin/bash

#SBATCH --account=katritch_223
# SBATCH --partition=gpu
# SBATCH --gpus-per-task=1
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --mail-user=mynguyen@usc.edu
#SBATCH --mail-type=end,fail
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=hsa
#SBATCH --output=hsa_%j.out

module purge
module load usc

module load gcc/11.3.0
module load cuda/11.5.1
export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/home1/mynguyen/.conda/envs/py310/bin:$PATH"
export PATH="/home1/mynguyen/.conda/envs/py310/lib/python3.10/site-packages/:$PATH"

python -u rf.py HSA