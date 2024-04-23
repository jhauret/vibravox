pyenv which python;
echo "Will run all STP trainings";
sleep 5;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=airborne.mouth_headworn.reference_microphone ++trainer.max_epochs=10;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=body_conducted.forehead.miniature_accelerometer ++trainer.max_epochs=10;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=body_conducted.in_ear.comply_foam_microphone ++trainer.max_epochs=10;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=body_conducted.in_ear.rigid_earpiece_microphone ++trainer.max_epochs=10;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=body_conducted.temple.contact_microphone ++trainer.max_epochs=10;
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=body_conducted.throat.piezoelectric_sensor ++trainer.max_epochs=10;
