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

## Available sensors

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6390fc80e6d656eb421bab69/P-_IWM3IMED5RBS3Lhydc.png" 
       style="max-width:50px; max-height:50px; width:auto; height:auto; border: solid 1px #CCC">
</p>



- 🟣:`headset_microphone` ( Not available for Bandwidth Extension as it is the reference mic )
- 🟡:`throat_microphone`
- 🟢: `forehead_accelerometer`
- 🔵:`rigid_in_ear_microphone`
- 🔴:`soft_in_ear_microphone`
- 🧊:`temple_vibration_pickup`
## Run some models

- [EBEN](https://github.com/jhauret/eben) for Bandwidth Extension  
  - Train and test on `speech_clean`, for recordings in a quiet environment:
    ```
    python run.py \
      lightning_datamodule=bwe \
      lightning_datamodule.sensor=throat_microphone \
      lightning_module=eben \
      lightning_module.generator.p=2 \
      +callbacks=[bwe_checkpoint] \
      ++trainer.check_val_every_n_epoch=15 \
      ++trainer.max_epochs=500
    ```
  - (Experimental) Train on `speech_clean` mixed with `speechless_noisy` and test on `speech_noisy`, for recordings in a noisy environment:
    ```
    python run.py \
      lightning_datamodule=noisybwe \
      lightning_datamodule.sensor=throat_microphone \
      lightning_module=eben \
      lightning_module.description=from_pretrained-throat_microphone \
      ++lightning_module.generator=dummy \
      ++lightning_module.generator._target_=vibravox.torch_modules.dnn.eben_generator.EBENGenerator.from_pretrained \
      ++lightning_module.generator.pretrained_model_name_or_path=Cnam-LMSSC/EBEN_throat_microphone \
      ++lightning_module.discriminator=dummy \
      ++lightning_module.discriminator._target_=vibravox.torch_modules.dnn.eben_discriminator.DiscriminatorEBENMultiScales.from_pretrained \
      ++lightning_module.discriminator.pretrained_model_name_or_path=Cnam-LMSSC/DiscriminatorEBENMultiScales_throat_microphone \
      +callbacks=[bwe_checkpoint] \
      ++callbacks.checkpoint.monitor=validation/torchmetrics_stoi/synthetic \
      ++trainer.check_val_every_n_epoch=15 \
      ++trainer.max_epochs=200
     ```

- [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli-v2) for Speech to Phoneme  
  - Train and test on `speech_clean`, for recordings in a quiet environment:  (weights initialized from [facebook/wav2vec2-base-fr-voxpopuli](https://huggingface.co/facebook/wav2vec2-base-fr-voxpopuli) )
  ```
  python run.py \
    lightning_datamodule=stp \
    lightning_datamodule.sensor=throat_microphone \
    lightning_module=wav2vec2_for_stp \
    lightning_module.optimizer.lr=1e-5 \
    ++trainer.max_epochs=10
  ```
  -  Train and test on `speech_noisy`, for recordings in a noisy environment:  (weights initialized from [vibravox_phonemizers](https://huggingface.co/Cnam-LMSSC/vibravox_phonemizers) )
    ```
  python run.py \
    lightning_datamodule=stp \
    lightning_datamodule.sensor=throat_microphone \
    lightning_datamodule.subset=speech_noisy \
    lightning_datamodule/data_augmentation=aggressive \
    lightning_module=wav2vec2_for_stp \
    lightning_module.wav2vec2_for_ctc.pretrained_model_name_or_path=Cnam-LMSSC/phonemizer_throat_microphone \
    lightning_module.optimizer.lr=1e-6 \
    ++trainer.max_epochs=30
  ```

- Test [ECAPA2](https://huggingface.co/Jenthe/ECAPA2) for Speaker Verification on `speech_clean`:
```
python run.py \
  lightning_datamodule=spkv \
  lightning_module=ecapa2 \
  logging=csv \
  ++trainer.limit_train_batches=0 \
  ++trainer.limit_val_batches=0
```