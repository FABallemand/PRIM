#!/bin/bash

#SBATCH --output=balle_reproduction_1.out
#SBATCH --error=balle_reproduction_1.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"

set -x
srun python3 -u balle_reproduction.py -m bmshj2018-hyperprior -d /home/ids/fallemand-24/PRIM/data/vimeo/vimeo_triplet --batch-size 8 -lr 1e-4 --save --cuda