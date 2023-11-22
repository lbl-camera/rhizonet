#!/bin/bash
#SBATCH -A zsordo
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%x.%j.out

echo "Starting at: $(date)"
module load python
conda activate rootNET
echo "In conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi
export SLURM_CPU_BIND="cores"
python predict2d.py 'setup_files/setup-predict2d.json'
echo "Done at : $(date)"