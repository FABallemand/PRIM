#!/bin/bash

#SBATCH --output=train_res/%j/%j_balle_reproduction.out
#SBATCH --error=train_res/%j/%j_balle_reproduction.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

mkdir train_res/$SLURM_JOB_ID

MODEL=bmshj2018-hyperprior

DATASET=/home/ids/fallemand-24/PRIM/data/vimeo/vimeo_triplet

CHECKPOINT=/home/ids/fallemand-24/PRIM/balle_reproduction/train_res/227892/checkpoint.pth.tar

eval "$(conda shell.bash hook)"

conda activate prim_env

set -x
# srun python3 -u balle_reproduction.py -m $MODEL -d $DATASET --num-workers 2 --epochs 1000000 --batch-size 8 -lr 1e-4 --cuda --savepath train_res/$SLURM_JOB_ID
srun python3 -u train.py -m $MODEL -d $DATASET --num-workers 2 --epochs 1000000 --batch-size 8 -lr 1e-4 --cuda --savepath train_res/$SLURM_JOB_ID --checkpoint $CHECKPOINT