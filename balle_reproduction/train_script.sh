#!/bin/bash

#SBATCH --output=balle_reproduction_1.out
#SBATCH --error=balle_reproduction_1.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"

set -x
srun python3 -u examples/train.py -m mbt2018-mean -d /path/to/image/dataset --batch-size 16 -lr 1e-4 --save --cuda