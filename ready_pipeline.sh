#!/bin/bash
module load python/3.7.4
echo "Conda ready to use"
echo "Please enter Y if chip-seq-pipeline virtual environment is already created, and N otherwise"
read n
if [ "$n" == "N" ]; then
bash /gpfs/data/rsingh47/masif/chip-seq-pipeline2/scripts/uninstall_conda_env.sh
bash /gpfs/data/rsingh47/masif/chip-seq-pipeline2/scripts/install_conda_env.sh mamba
fi
echo "If you did not enter N, and the pipeline environment is not installed the script will error on the next step." 
conda activate encode-chip-seq-pipeline
conda install tbb=2020.2
pip3 install caper
pip3 install croo
caper init local
echo "Please Head to ~/.caper/default.conf and edit the local path, set it to /gpfs/data/rsingh47/masif/caper_output, when done press Y"
read y
if [ "$y" == "Y" ]; then
export PATH=$PATH:/gpfs/data/rsingh47/masif/sratoolkit.2.11.1-centos_linux64/bin
source /gpfs/data/rsingh47/masif/ChromAge/ChromAge_venv/bin/activate
python3 /gpfs/data/rsingh47/masif/ChromAge/pre-process.py
fi
echo "Please enter Y, only then will the script run the pipeline"
