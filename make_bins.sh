#!/bin/bash

#SBATCH -n 100
#SBATCH --mem=100G
#SBATCH -t 48:00:00
#SBATCH -o H3K27ac-binnig.out

module load python/3.7.4

source ChromAge_venv/bin/activate

python3 raw_data_processing.py

deactivate 
