#!/bin/bash

#SBATCH --output=test_res/%j/%j_test.out
#SBATCH --error=test_res/%j/%j_test.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=3090

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

eval "$(conda shell.bash hook)"

conda activate prim_env

set -x
srun python3 -u test_lambda.py