#!/bin/bash

echo "Submit all BWE trainings to be run in parallel";

SENSORS=(
    "body_conducted.forehead.miniature_accelerometer"
    "body_conducted.in_ear.comply_foam_microphone"
    "body_conducted.in_ear.rigid_earpiece_microphone"
    "body_conducted.temple.contact_microphone"
    "body_conducted.throat.piezoelectric_sensor"
)

declare -A EBEN_MAPPING_P
EBEN_MAPPING_P=(
    ["body_conducted.forehead.miniature_accelerometer"]=4
    ["body_conducted.in_ear.comply_foam_microphone"]=2
    ["body_conducted.in_ear.rigid_earpiece_microphone"]=2
    ["body_conducted.temple.contact_microphone"]=1
    ["body_conducted.throat.piezoelectric_sensor"]=1
)

for SENSOR in "${SENSORS[@]}"; do
    (

    #SBATCH --job-name="baseline_$DATA_MODULE"
    #SBATCH --output="$JOB_NAME.out"
    #SBATCH --error="$JOB_NAME.err"
    #SBATCH --constraint=v100-16g
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=10
    #SBATCH --time=00:30:00
    #SBATCH --qos=qos_gpu-dev
    #SBATCH --hint=nomultithread
    #SBATCH --account=lbo@v100

    # Activate required environment
    module purge
    conda deactivate
    module load pytorch-gpu/py3/2.2.0

    # Set environment variables
    export HF_DATASETS_CACHE=$WORK/huggingface_cache/datasets
    export HF_DATASETS_OFFLINE=1

    # Set the corresponding lightning_module based on the lightning_datamodule
    P=${EBEN_MAPPING_P[SENSOR]}

    # Enable command echoing
    set -x

    # Execute the job
    srun python -u run.py lightning_datamodule=bwe lightning_datamodule.sensor="$SENSOR" lightning_module=eben lightning_module.generator.p="$P" ++trainer.check_val_every_n_epoch=15
    ) &
done

# Wait for all background jobs to finish
wait
