# Vibravox
Speech to Phoneme, Bandwidth Extension and Speaker Identification using the Vibravox dataset.

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.1-792ee5?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>

## Requirements
```pip install -r requirements.txt```

## Train some models

- Train EBEN for Bandwidth Extension  
```python train.py lightning_datamodule=bwe lightning_module=eben```


- Train wav2vec2 for Speech to Phoneme  
```python train.pylightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=airborne.mouth_headworn.reference_microphone ++trainer.max_epochs=10```

