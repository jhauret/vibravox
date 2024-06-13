#!/bin/bash

#SBATCH --job-name=stp_array_job
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=lbo@v100
#SBATCH --array=1-6

module purge
conda deactivate

module load pytorch-gpu/py3/2.2.0

export HF_DATASETS_CACHE=$WORK/huggingface_cache/datasets
export HF_DATASETS_OFFLINE=1

# Specify the path to the config file
array_config=./configs/slurm_array/stp.txt

# Extract values of the job
sensor=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $array_config)

set -x
srun python -u run.py lightning_datamodule=stp lightning_datamodule.sensor="$sensor" lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=headset_microphone ++trainer.max_epochs=10