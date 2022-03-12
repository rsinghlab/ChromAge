#!/bin/bash

#SBATCH -n 32
#SBATCH --mem=100G
#SBATCH -t 2:00:00
#SBATCH -o H3K4me3-post-process.out

module load python/3.7.4
source ChromAge_venv/bin/activate

python3 /users/masif/data/masif/ChromAge/simple_nn.py

deactivate
