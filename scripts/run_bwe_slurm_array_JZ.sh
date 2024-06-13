#!/bin/bash

#SBATCH --job-name=bwe_array_job
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=99:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --hint=nomultithread
#SBATCH --account=lbo@v100
#SBATCH --array=1-5

module purge
conda deactivate

module load pytorch-gpu/py3/2.2.0

export HF_DATASETS_CACHE=$WORK/huggingface_cache/datasets
export HUGGINGFACE_HUB_CACHE=$WORK/huggingface_cache/hub
export HF_DATASETS_OFFLINE=1

# Specify the path to the config file
array_config=./configs/slurm_array/bwe.txt

# Extract values of the job
sensor=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $array_config)
p=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $array_config)
q=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $array_config)
min_channels=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $array_config)

set -x
srun python -u run.py lightning_datamodule=bwe lightning_datamodule.sensor="$sensor" lightning_module=eben lightning_module.generator.p="$p" lightning_module.discriminator.q="$q" lightning_module.discriminator.min_channels="$min_channels" ++trainer.check_val_every_n_epoch=15 ++trainer.max_epochs=600
