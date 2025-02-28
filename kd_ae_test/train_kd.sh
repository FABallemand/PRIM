#!/bin/bash

#SBATCH --output=train_res/%j/%j_train_kd.out
#SBATCH --error=train_res/%j/%j_train_kd.err
#SBATCH --time=99:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

mkdir train_res/$SLURM_JOB_ID

DATASET=/home/ids/fallemand-24/PRIM/data/vimeo/vimeo_triplet

# TEACHER_CHECKPOINT=/home/ids/fallemand-24/PRIM/kd_ae/train_res/243807/checkpoint_best.pth.tar
TEACHER_CHECKPOINT=/home/ids/fallemand-24/PRIM/kd_ae/train_res/274518/checkpoint_best.pth.tar

eval "$(conda shell.bash hook)"

conda activate prim_env

set -x
srun python3 -u train_kd.py -d $DATASET --num-workers 2 --epochs 1000 --batch-size 16 -lr 1e-4 --cuda --savepath train_res/$SLURM_JOB_ID --teacher-checkpoint $TEACHER_CHECKPOINT