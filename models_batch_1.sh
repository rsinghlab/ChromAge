#!/bin/bash

#SBATCH -n 30
#SBATCH --mem=100G
#SBATCH -t 250:00:00
#SBATCH -o simple-nn-autoencoder_run_1.out

module load python/3.7.4
source ChromAge_venv/bin/activate

python3 /users/masif/data/masif/ChromAge/simple_nn.py

deactivate
