#!/bin/bash

#SBATCH -n 30
#SBATCH --mem=100G
#SBATCH -t 600:00:00
#SBATCH -o pipeline-run-%j.out

sh ~/data/masif/ChromAge/ready_pipeline.sh
