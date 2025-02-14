#!/bin/bash

#SBATCH --job-name=noisy_bwe_array_job
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
export HF_HUB_OFFLINE=1

# Specify the path to the config file
array_config=./configs/slurm_array/bwe.txt

# Extract values of the job
sensor=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $array_config)
p=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $array_config)

set -x
srun python -u run.py \
  lightning_datamodule=noisybwe \
  lightning_datamodule.sensor="$sensor" \
  lightning_module=eben \
  lightning_module.description=from_pretrained-"$sensor" \
  ++lightning_module.generator=dummy \
  ++lightning_module.generator._target_=vibravox.torch_modules.dnn.eben_generator.EBENGenerator.from_pretrained \
  ++lightning_module.generator.pretrained_model_name_or_path=Cnam-LMSSC/EBEN_"$sensor" \
  ++lightning_module.discriminator=dummy \
  ++lightning_module.discriminator._target_=vibravox.torch_modules.dnn.eben_discriminator.DiscriminatorEBENMultiScales.from_pretrained \
  ++lightning_module.discriminator.pretrained_model_name_or_path=Cnam-LMSSC/DiscriminatorEBENMultiScales_"$sensor" \
  +callbacks=[bwe_checkpoint] \
  ++callbacks.checkpoint.monitor=validation/torchmetrics_stoi/synthetic \
  ++trainer.check_val_every_n_epoch=15 \
  ++trainer.max_epochs=300