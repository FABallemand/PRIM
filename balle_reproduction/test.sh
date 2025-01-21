#!/bin/bash

#SBATCH --output=test_res/%j/%j_balle_reproduction.out
#SBATCH --error=test_res/%j/%j_balle_reproduction.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

mkdir test_res/$SLURM_JOB_ID

MODEL="bmshj2018-hyperprior"

DATASET=/home/ids/fallemand-24/PRIM/data/kodak

CHECKPOINT=/home/ids/fallemand-24/PRIM/balle_reproduction/train_res/224584/checkpoint.pth.tar

eval "$(conda shell.bash hook)"

conda activate prim_env

set -x
srun python3 -u test.py checkpoint -a $MODEL --dataset $DATASET -p $CHECKPOINT --cuda -d test_res/$SLURM_JOB_ID -o $SLURM_JOB_ID.json