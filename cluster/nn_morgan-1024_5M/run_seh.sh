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
#SBATCH --job-name=seh
#SBATCH --output=seh_%j.out

module purge
module load usc

module load gcc/11.3.0
module load cuda/11.8.0
export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/home1/mynguyen/.conda/envs/py310/bin:$PATH"
export PATH="/home1/mynguyen/.conda/envs/py310/lib/python3.10/site-packages/:$PATH"

cp='../nn_morgan-1024_1M/checkpoints_NN4_sEH_10_trainsize-1000000_posrate-0.05_resampling-False_reweighting-False/checkpoint_epoch_30.pth'
#cp='../nn_morgan-1024_1M/checkpoints_NN5_sEH_10_trainsize-1000000_posrate-0.05_resampling-False_reweighting-False/checkpoint_epoch_35.pth'

python -u nn.py sEH -model_name NN4 -checkpoint $cp -model_index 10-v1
#python -u nn_v2.py sEH -model_name NN5 -checkpoint $cp -model_index 10
