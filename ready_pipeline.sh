#!/bin/bash

#SBATCH -n 30
#SBATCH --mem=100G
#SBATCH -t 600:00:00
#SBATCH -o pipeline-run-%j.out
#SBATCH -A cbc-condo

module load python/3.7.4
source /gpfs/home/masif/data/masif/miniconda3/etc/profile.d/conda.sh
echo "Conda ready to use"
#echo "Please enter Y if chip-seq-pipeline virtual environment is already created, and N otherwise"
#read n
#if [ "$n" == "N" ]; then
#bash /gpfs/data/rsingh47/masif/chip-seq-pipeline2/scripts/uninstall_conda_env.sh
#bash /gpfs/data/rsingh47/masif/chip-seq-pipeline2/scripts/install_conda_env.sh mamba
#fi
#echo "If you did not enter N, and the pipeline environment is not installed the script will error on the next step." 
conda activate encode-chip-seq-pipeline
#conda install tbb=2020.2
#cp /gpfs/home/masif/data/masif/miniconda3/envs/encode-chip-seq-pipeline/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_linux_gnu.py /gpfs/home/masif/data/masif/miniconda3/envs/encode-chip-seq-pipeline/lib/python3.7/_sysconfigdata_x86_64_conda_linux_gnu.py
#pip3 uninstall caper
#pip3 install caper
#pip3 install croo
#caper init local
#echo "Please Head to ~/.caper/default.conf and edit the local path, set it to /gpfs/data/rsingh47/masif/caper_output, when done press Y"
#read y
#if [ "$y" == "Y" ]; then
export PATH=$PATH:/gpfs/data/rsingh47/masif/sratoolkit.2.11.1-centos_linux64/bin
source /gpfs/data/rsingh47/masif/ChromAge/ChromAge_venv/bin/activate
python3 /gpfs/data/rsingh47/masif/ChromAge/pre-process.py
deactivate
#fi

#rm -f /gpfs/data/rsingh47/masif/ChromAge/cromwell.out
#rm -r /gpfs/data/rsingh47/masif/ChromAge/chip
#rm -r /gpfs/data/rsingh47/masif/ChromAge/cromwell-workflow-logs/
#rm -r /gpfs/data/rsingh47/masif/caper_output/chip
