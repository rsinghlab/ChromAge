#!/bin/bash

#SBATCH -n 96
#SBATCH --mem=100G
#SBATCH -t 400:00:00
#SBATCH -o simple-nn-autoencoder_run.out
#SBATCH -A cbc-condo

module load python/3.7.4
source ChromAge_venv/bin/activate

python3 /users/masif/data/masif/ChromAge/simple_nn.py

deactivate
