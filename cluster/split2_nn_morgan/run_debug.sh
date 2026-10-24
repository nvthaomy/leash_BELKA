#!/bin/bash

module load gcc/11.3.0
module load cuda/11.8.0
export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/home1/mynguyen/.conda/envs/py310/bin:$PATH"
export PATH="/home1/mynguyen/.conda/envs/py310/lib/python3.10/site-packages/:$PATH"

split=posrate-0.050_size-0.005M

model=checkpoints_posrate-0.082_size-5.0M_NN5_00_1024bits_invar1024bits_resampling-False_reweighting-False/model_fold-0.pth
python -u nn_v3.5.py  -gpu -split $split -model_name NN5 -load_model $model -fp_invar_size 1024