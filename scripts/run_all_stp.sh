pyenv which python;
echo "Will run all STP trainings sequentially";
sleep 5;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=headset_microphone ++trainer.max_epochs=10 ++lightning_module.push_to_hub_after_testing=True;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=forehead_accelerometer ++trainer.max_epochs=10 ++lightning_module.push_to_hub_after_testing=True;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=soft_in_ear_microphone ++trainer.max_epochs=10 ++lightning_module.push_to_hub_after_testing=True;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=rigid_in_ear_microphone ++trainer.max_epochs=10 ++lightning_module.push_to_hub_after_testing=True;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=temple_vibration_pickup ++trainer.max_epochs=10 ++lightning_module.push_to_hub_after_testing=True;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=throat_microphone ++trainer.max_epochs=10 ++lightning_module.push_to_hub_after_testing=True;
