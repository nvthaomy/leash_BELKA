#!/bin/bash

# SBATCH --account=katritch_223
# SBATCH --partition=gpu

#SBATCH --account=gati_564
#SBATCH --partition=ruska
#SBATCH --mem=350G
#SBATCH --gpus-per-task=1

# SBATCH --partition=main
# SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --mail-user=mynguyen@usc.edu
#SBATCH --mail-type=end,fail
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=random
#SBATCH --output=log_%j.out

# SBATCH --nodelist=b04-05

module purge
module load usc

module load gcc/11.3.0
module load cuda/11.8.0
export OPENMM_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/home1/mynguyen/.conda/envs/py310/bin:$PATH"
export PATH="/home1/mynguyen/.conda/envs/py310/lib/python3.10/site-packages/:$PATH"

# prot=sEH
# path=../data/split2/train_folds.parquet
# split=${prot}_posrate-0.050_size-5.0M
# model=../random_split_nn_morgan-1024/reference/checkpoints_5000000_posrate-0.05_NN4_${prot}_00_1024bits_resampling-False_reweighting-False/model_fold-6.pth

# python -u nn_v3.5.py -protein $prot -load_model $model -gpu -process_train -train_path $path -split $split

split=posrate-0.041_size-10.0M
#posrate-0.082_size-5.0M
#posrate-0.041_size-10.0M 
#posrate-0.027_size-15.0M

# model=checkpoints_posrate-0.082_size-5.0M_NN4_00_1024bits_invar0bits_resampling-False_reweighting-False/model_fold-5.pth
# python -u nn_v3.5.py  -gpu -split $split -model_name NN4 -load_model $model 

model=checkpoints_posrate-0.041_size-10.0M_NN5_00_1024bits_invar1024bits_resampling-False_reweighting-False/model_fold-0.pth
python -u nn_v3.5.py  -gpu -split $split -model_name NN5 -load_model $model -fp_invar_size 1024