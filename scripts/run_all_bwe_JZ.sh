#!/bin/bash

#SBATCH --job-name=array_job
#SBATCH --output=array_job%j.out
#SBATCH --error=array_job%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:01:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --account=lbo@v100
#SBATCH --array=1-5

module purge
conda deactivate

module load pytorch-gpu/py3/2.2.0

export HF_DATASETS_CACHE=$WORK/huggingface_cache/datasets
export HF_DATASETS_OFFLINE=1

# Specify the path to the config file
array_config=./configs/slurm_array/bwe.txt

# Extract values of the job
sensor=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $array_config)
p=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $array_config)

set -x
srun python -u run.py lightning_datamodule=bwe lightning_datamodule.sensor="$sensor" lightning_module=eben lightning_module.generator.p="$p" ++trainer.check_val_every_n_epoch=15
