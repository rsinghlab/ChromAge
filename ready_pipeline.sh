#!/bin/bash
module load python/3.7.4
echo "Activating Conda environment"
source /gpfs/home/masif/data/masif/galaxy/database/dependencies/_conda/bin/activate
conda deactivate
echo "Conda ready to use"
echo "Please enter Y if chip-seq-pipeline virtual environment is already created, and N otherwise"
read n
if [ "$n" == "N" ]; then
bash /gpfs/home/masif/data/masif/chip-seq-pipeline2/scripts/uninstall_conda_env.sh
bash /gpfs/home/masif/data/masif/chip-seq-pipeline2/scripts/install_conda_env.sh mamba
fi
echo "If you did not enter N, and the pipeline environment is not installed the script will error on the next step."
cp /gpfs/home/masif/data/masif/galaxy/database/dependencies/_conda/envs/encode-chip-seq-pipeline/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_linux_gnu.py /gpfs/home/masif/data/masif/galaxy/database/dependencies/_conda/envs/encode-chip-seq-pipeline/lib/python3.7/_sysconfigdata_x86_64_conda_linux_gnu.py  
conda activate encode-chip-seq-pipeline
conda install tbb=2020.2
pip3 install caper
pip3 install croo
caper init local
echo "Please Head to ~/.caper/default.conf and edit the local path, set it to /gpfs/home/masif/data/masif/chip-seq-pipeline2, when done press Y"
read y
if [ "$y" == "Y" ]; then
export PATH=$PATH:/gpfs/home/masif/data/masif/sratoolkit.2.11.1-centos_linux64/bin
source /gpfs/home/masif/data/masif/ChromAge/ChromAge_venv/bin/activate
python3 /gpfs/home/masif/data/masif/ChromAge/pre-process.py
fi
echo "Please enter Y, only then will the script run the pipeline"
