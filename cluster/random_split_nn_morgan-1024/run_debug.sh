#!/bin/bash

module load gcc/11.3.0
module load cuda/11.8.0
export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/home1/mynguyen/.conda/envs/py310/bin:$PATH"
export PATH="/home1/mynguyen/.conda/envs/py310/lib/python3.10/site-packages/:$PATH"

cp=''

split_name=HSA_posrate-0.050_size-5.0M

python -u nn_v3.py -model_name NN4 -model_index 00 -fold 6 -split_name $split_name -protein HSA -process_train -fp_size 512 -fp_invar_size 512
