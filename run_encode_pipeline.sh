#!/bin/bash

#SBATCH -n 6
#SBATCH --mem=16G
#SBATCH -t 300:00:00

sh ~/data/masif/ChromAge/ready_pipeline.sh
