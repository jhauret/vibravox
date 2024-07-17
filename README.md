<div align="center">

<p align="center">
  <img src="./logo.png" style="object-fit:contain; width:250px; height:250px; border: solid 1px #CCC">
</p>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.2-792ee5?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-🐉 hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>
<a href="https://huggingface.co/datasets"><img alt="HuggingFace Datasets" src="https://img.shields.io/badge/datasets 2.19-yellow?style=for-the-badge&logo=huggingface&logoColor=white"></a>



Speech to Phoneme, Bandwidth Extension and Speaker Verification using the Vibravox dataset.



</div>

## Resources:

- **📝**: The paper related to this project is available on arXiv on [this link](https://arxiv.org/abs/2407.11828).
- **🤗**: The dataset used in this project is hosted by Hugging Face. You can access it [here](https://huggingface.co/datasets/Cnam-LMSSC/vibravox).  
- **🌐**: For more information about the project, visit our [project page](https://vibravox.cnam.fr/).
- **🏆**: Explore Leaderboards on [Papers With Code](https://paperswithcode.com/paper/vibravox-a-dataset-of-french-speech-captured).

## Requirements
```pip install -r requirements.txt```

## Run some models

- Train [EBEN](https://github.com/jhauret/eben) for Bandwidth Extension  
```
python run.py lightning_datamodule=bwe lightning_datamodule.sensor=throat_microphone lightning_module=eben  ++trainer.check_val_every_n_epoch=15 ++trainer.max_epochs=500
```

- Train [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli-v2) for Speech to Phoneme  
```
python run.py lightning_datamodule=stp lightning_datamodule.sensor=headset_microphone lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 ++trainer.max_epochs=10
```

- Test [ECAPA2](https://huggingface.co/Jenthe/ECAPA2) for Speaker Verification
```
python run.py lightning_datamodule=spkv lightning_module=ecapa2 logging=csv ++trainer.limit_train_batches=0 ++trainer.limit_val_batches=0
```
