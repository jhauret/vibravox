<div align="center">

<p align="center">
  <img src="./logo.png" style="object-fit:contain; width:250px; height:250px; border: solid 1px #CCC">
</p>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.2-792ee5?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-🐉 hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>
<a href="https://huggingface.co/datasets"><img alt="HuggingFace Datasets" src="https://img.shields.io/badge/-🤗 datasets 2.19-yellow?style=for-the-badge&logo=huggingface&logoColor=white"></a>



Speech to Phoneme, Bandwidth Extension and Speaker Verification using the Vibravox dataset.



</div>

## Resources:

- **📝**: The paper related to this project is yet to be published. Stay tuned for updates!  
- **🤗**: The dataset used in this project is hosted by Hugging Face. You can access it [here](https://huggingface.co/datasets/Cnam-LMSSC/vibravox).  
- **🌐**: For more information about the project, visit our [project page](https://vibravox.cnam.fr/).

## Requirements
```pip install -r requirements.txt```

## Train/Test some models

- Train [EBEN](https://github.com/jhauret/eben) for Bandwidth Extension  
```
python run.py lightning_datamodule=bwe lightning_module=eben
```

- Train [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli-v2) for Speech to Phoneme  
```
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=airborne.mouth_headworn.reference_microphone ++trainer.max_epochs=10
```

- **Test** [ECAPA2](https://huggingface.co/Jenthe/ECAPA2) for Speaker Verification
```
python run.py lightning_datamodule=spkv lightning_module=ecapa2 ++trainer.limit_train_batches=0
```
