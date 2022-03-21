#!/bin/bash

#SBATCH -n 16
#SBATCH --mem=45G
#SBATCH -t 500:00:00
#SBATCH -o H3K27ac-search.out
#SBATCH -A cbc-condo

module load python/3.7.4
source ChromAge_venv/bin/activate

python3 /users/masif/data/masif/ChromAge/simple_nn.py

deactivate
