#!/bin/bash

#SBATCH --output=results/balle_reproduction_%j.out
#SBATCH --error=results/balle_reproduction_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

mkdir results/$SLURM_JOB_ID

MODEL=bmshj2018-hyperprior

DATASET=/home/ids/fallemand-24/PRIM/data/vimeo/vimeo_triplet

eval "$(conda shell.bash hook)"

conda activate balle_reproduction

set -x
srun python3 -u balle_reproduction.py -m $MODEL -d $DATASET --num-workers 2 --epochs 1000000 --batch-size 8 -lr 1e-4 --cuda --savepath results/$SLURM_JOB_ID