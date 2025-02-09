#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --output=baseline%j.out
#SBATCH --error=baseline%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=99:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --hint=nomultithread
#SBATCH --account=lbo@v100

module purge                                          # nettoyer les modules herites par defaut
conda deactivate                                      # desactiver les environnements herites par defaut

module load pytorch-gpu/py3/2.2.0                     # charger les modules

export HF_DATASETS_CACHE=$WORK/huggingface_cache/datasets
export HUGGINGFACE_HUB_CACHE=$WORK/huggingface_cache/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

set -x                                                # activer l'echo des commandes
srun python -u run.py lightning_datamodule=bwe lightning_module=eben ++trainer.check_val_every_n_epoch=15 +callbacks=bwe_checkpoint
