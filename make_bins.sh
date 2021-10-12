#!/bin/bash

#SBATCH -n 100
#SBATCH --mem=100G
#SBATCH -t 180:00:00
#SBATCH -o my-output-%j.out

source ChromAge_venv/bin/activate

python3 raw_data_processing.py

deactivate 
