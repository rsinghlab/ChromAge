#!/bin/bash

cd /gpfs/data/rsingh47/masif/caper_output/bigWigs/ 
for e in *.fc.signal.bigwig ; do
echo $e
mv $e /gpfs/data/rsingh47/masif/ChromAge/GEO_histone_data/H3K4me1/raw_data/fc
done

cd /gpfs/data/rsingh47/masif/caper_output/bigWigs/
for e in *.pval.signal.bigwig ; do
echo $e
mv $e /gpfs/data/rsingh47/masif/ChromAge/GEO_histone_data/H3K4me1/raw_data/pval
done

mv /gpfs/data/rsingh47/masif/caper_output/qcReports/ /gpfs/data/rsingh47/masif/ChromAge/GEO_histone_data/H3K4me1/
