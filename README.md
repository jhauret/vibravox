# Vibravox
Speech to Phoneme, Bandwidth Extension and Speaker Identification using the Vibravox dataset

## Requirements
```pip install -r requirements.txt```

## Train some models

- Train EBEN for Bandwidth Extension  
```python train.py lightning_datamodule=bwe lightning_module=eben```


- Train wav2vec2 for Speech to Phoneme  
```python train.py lightning_datamodule=stp lightning_module=wav2vec2_for_stp```

