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

# By default metric is MSE
# QUALITY  1      2      3      4      5      6      7      8
# MSE 0.0018 0.0035 0.0067 0.0130 0.0250 0.0483 0.0932 0.1800
QUALITY=8
LAMBDA=0.1800

eval "$(conda shell.bash hook)"

conda activate prim_env

set -x
srun python3 -u train.py -m $MODEL -d $DATASET --num-workers 2 --epochs 65 --batch-size 16 -lr 1e-4 --cuda --savepath train_res/$SLURM_JOB_ID --lambda $LAMBDA --quality $QUALITY