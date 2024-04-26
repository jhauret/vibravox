<div align="center">

![vibravox_logo](https://github.com/jhauret/vibravox/assets/53187038/549b8e16-208f-4321-8693-0fda996b9a78) 

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.1-792ee5?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>

Speech to Phoneme, Bandwidth Extension and Speaker Verification using the Vibravox dataset.



</div>

## Resources:

- **📝**: The paper related to this project is yet to be published. Stay tuned for updates!  
- **🤗**: The dataset used in this project is hosted by Hugging Face. You can access it [here](https://huggingface.co/datasets/Cnam-LMSSC/vibravox).  
- **🌐**: For more information about the project, visit our [project page](https://vibravox.cnam.fr/).

## Requirements
```pip install -r requirements.txt```

## Train some models

- Train EBEN for Bandwidth Extension  
```
python run.py lightning_datamodule=bwe lightning_module=eben
```

- Train [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli-v2) for Speech to Phoneme  
```
python run.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp lightning_module.optimizer.lr=1e-5 lightning_datamodule.sensor=airborne.mouth_headworn.reference_microphone ++trainer.max_epochs=10
```

- Train [titanet](https://huggingface.co/nvidia/speakerverification_en_titanet_large) for Speaker Verification
```
python run.py lightning_datamodule=spk_verif lightning_module=titanet
```
