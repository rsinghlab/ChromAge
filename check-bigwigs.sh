#!/bin/bash
module load python/3.7.4
source /gpfs/data/rsingh47/masif/ChromAge/ChromAge_venv/bin/activate

cd /gpfs/data/rsingh47/masif/caper_output/bigWigs/

part1="/gpfs/data/rsingh47/masif/caper_output/bigWigs/"

for d in * ; do
python3 check-wigs.py "$part1$d"
done

