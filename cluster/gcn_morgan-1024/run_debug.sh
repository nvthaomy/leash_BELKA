#!/bin/bash

module load gcc/11.3.0
module load cuda/11.8.0
export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/home1/mynguyen/.conda/envs/py310/bin:$PATH"
export PATH="/home1/mynguyen/.conda/envs/py310/lib/python3.10/site-packages/:$PATH"

path=../data/split2/train_folds.parquet
model=checkpoints_posrate-0.050_size-0.005M_FPGCN_00_1024bits_invar0bits_resampling-False_reweighting-False/model_fold-0.pth
split=posrate-0.050_size-0.005M
#posrate-0.082_size-5.0M
python -u nn_v4.py -split_name $split -train_path $path  -model_name GCN -gpu -model_index 00
