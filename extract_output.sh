#!/bin/bash
cd /gpfs/data/rsingh47/masif/caper_output/chip

part1="/gpfs/data/rsingh47/masif/caper_output/chip/"
part2="signal/rep1/"
part3="qc"

conda activate encode-chip-seq-pipeline

for d in */ ; do
echo $d
cd /gpfs/data/rsingh47/masif/caper_output/chip/$d
croo metadata.json
mv "$part1$d$part3" /gpfs/data/rsingh47/masif/caper_output/qcReports/$d
cd "$part1$d$part2"
for e in *.bigwig ; do
echo $e
mv $e /gpfs/data/rsingh47/masif/caper_output/bigWigs
done
rm -r /gpfs/data/rsingh47/masif/caper_output/chip/$d
done