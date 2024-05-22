"""
This script is the entry point for training and testing the model.
It instantiates all necessary modules, trains the model and tests it.
"""

import os
from typing import List
import warnings
import hydra
import torch
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    seed_everything,
    Trainer,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig

from torchmetrics import MetricCollection


@hydra.main(
    config_path="../../configs",
    config_name="spkv_test.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """
    Instantiate all necessary module and test the model.

    Args:
        cfg (DictConfig): Hydra configuration object, passed in by the @hydra.main decorator
    """

    # Instantiate LightningDataModule
    lightning_datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.lightning_datamodule
    )

    # Instantiate LightningModule
    metrics: MetricCollection = MetricCollection(
        dict(hydra.utils.instantiate(cfg.metrics))
    )
    lightning_module: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        metrics=metrics,
    )

    # Instantiate Trainer
    logger: TensorBoardLogger = hydra.utils.instantiate(cfg.logging.logger)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, _convert_="partial"
    )

    # Test the model
    trainer.test(lightning_module, datamodule=lightning_datamodule)


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
