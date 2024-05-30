#!/bin/bash

#SBATCH --job-name=spkv_array_job
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=lbo@v100
#SBATCH --array=1-144

module purge
conda deactivate

module load pytorch-gpu/py3/2.2.0

export HF_DATASETS_CACHE=$WORK/huggingface_cache/datasets
export HF_DATASETS_OFFLINE=1

# Specify the path to the config file
array_config=./configs/slurm_array/spkv.txt

# Extract values of the job
dataset_name=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $array_config)
split=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $array_config)
sensor_a=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $array_config)
sensor_b=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $array_config)
pairs=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $array_config)

set -x
srun python -u run.py lightning_datamodule=spkv lightning_module=ecapa2 lightning_datamodule.dataset_name="$dataset_name"  lightning_datamodule.split="$split" lightning_datamodule.sensor_a="$sensor_a" lightning_datamodule.sensor_b="$sensor_b" lightning_datamodule.pairs="$pairs" ++trainer.limit_train_batches=0
