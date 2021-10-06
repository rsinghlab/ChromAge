#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00

sh ~/data/masif/ChromAge/ready_pipeline.sh
