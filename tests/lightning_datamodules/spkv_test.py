"""
This script is the entry point for training and testing the model.
It instantiates all necessary modules, trains the model and tests it.
"""


import os
import warnings
import hydra
import torch
from lightning import (
    LightningDataModule,
    seed_everything,
)
from omegaconf import DictConfig



@hydra.main(
    config_path="../../configs",
    config_name="test_spkv.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """
    Instantiate all necessary modules, train and test the model.

    Args:
        cfg (DictConfig): Hydra configuration object, passed in by the @hydra.main decorator
    """

    # Instantiate LightningDataModule
    lightning_datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.lightning_datamodule
    )
    # Test the DataModule for test set :

    print('#############################################################,\n')
    print("Getting instances of the test set using CombineDataloder : \n")

    lightning_datamodule.setup(stage = 'test')

    test_iterator = iter(lightning_datamodule.test_dataloader())

    batch = next(test_iterator)

    print("\n Combined Batch from CombinedDataloader : \n ")
    print(batch)
    print("\n Keys from Dataloader 'sensor_a' :")
    print(batch[0]['sensor_a'].keys())
    print(" \n Values from Dataloader 'sensor_a' : \n " )
    print((batch[0]['sensor_a']['audio']))
    print((batch[0]['sensor_a']['speaker_id']))
    print((batch[0]['sensor_a']['sentence_id']))
    print((batch[0]['sensor_a']['gender']))
    print((batch[0]['sensor_a']['sensor']))
    print("\n Keys from Dataloader 'sensor_b' : \n")
    print(batch[0]['sensor_b'].keys())
    print("\n Values from Dataloader 'sensor_b' : \n")
    print((batch[0]['sensor_b']['audio']))
    print((batch[0]['sensor_b']['speaker_id']))
    print((batch[0]['sensor_b']['sentence_id']))
    print((batch[0]['sensor_b']['gender']))
    print((batch[0]['sensor_b']['sensor']))


    print('#############################################################,\n')
    print("Getting instances of the train set using standard Dataloader : \n")

    lightning_datamodule.setup(stage='fit')
    train_iterator = iter(lightning_datamodule.train_dataloader())

    batch = next(train_iterator)

    print("\n Batch from Train Dataloder : \n ")
    print(batch)
    print("\n Keys from batch :")
    print(batch.keys())
    print(" \n Values from Train Dataloader : \n ")
    print((batch['audio']))
    print((batch['speaker_id']))
    print((batch['sentence_id']))
    print((batch['gender']))
    print((batch['sensor']))

    batch = next(train_iterator)

    print("\n Second Batch from Train Dataloader : \n ")
    print(batch)
    print("\n Keys from batch :")
    print(batch.keys())
    print(" \n Values from Validation Dataloader : \n ")
    print((batch['audio']))
    print((batch['speaker_id']))
    print((batch['sentence_id']))
    print((batch['gender']))
    print((batch['sensor']))


    print('#############################################################,\n')
    print("Getting instances of the validation set using standard Dataloader : \n")

    lightning_datamodule.setup(stage='fit')
    val_iterator = iter(lightning_datamodule.val_dataloader())

    batch = next(val_iterator)

    print("\n Batch from Validation Dataloader : \n ")
    print(batch)
    print("\n Keys from batch :")
    print(batch.keys())
    print(" \n Values from Validation Dataloader : \n ")
    print((batch['audio']))
    print((batch['speaker_id']))
    print((batch['sentence_id']))
    print((batch['gender']))
    print((batch['sensor']))

    batch = next(val_iterator)

    print("\n Second Batch from Validation Dataloder : \n ")
    print(batch)
    print("\n Keys from batch :")
    print(batch.keys())
    print(" \n Values from Validation Dataloader : \n ")
    print((batch['audio']))
    print((batch['speaker_id']))
    print((batch['sentence_id']))
    print((batch['gender']))
    print((batch['sensor']))



def setup_environment():
    """
    Setup environment for training
    """

    warnings.filterwarnings("ignore")

    # Set environment variables for full trace of errors
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Enable CUDNN backend
    torch.backends.cudnn.enabled = True

    # Enable CUDNN benchmarking to choose the best algorithm for every new input size
    # e.g. for convolutional layers chose between Winograd, GEMM-based, or FFT algorithms
    torch.backends.cudnn.benchmark = True

    # Sets seeds for numpy, torch and python.random for reproducibility
    seed_everything(42, workers=True)


if __name__ == "__main__":

    setup_environment()
    main()
