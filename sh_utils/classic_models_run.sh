#!/bin/bash

#SBATCH -n 100
#SBATCH --mem=100G
#SBATCH -t 100:00:00
#SBATCH -o classic_models.out

module load python/3.7.4

source ChromAge_venv/bin/activate

python3 python_utils/classical_models.py

deactivate 
