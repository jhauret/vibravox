import os
from typing import List, Tuple

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
    config_path="./configs",
    config_name="train_bwe.yaml",
    version_base="1.3",
)
def instantiate(
    cfg: DictConfig,
) -> Tuple[Trainer, LightningModule, LightningDataModule]:
    """
    Instantiate all necessary modules for training

    Args:
        cfg (DictConfig): Hydra configuration object
    Returns
        Tuple[Trainer, LightningModule, LightningDataModule]: Trainer, LightningModule, LightningDataModule
    """
    # Instantiate LightningDataModule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.lightning_datamodule)

    # Instantiate Metrics
    metrics: MetricCollection = MetricCollection(dict(hydra.utils.instantiate(cfg.metrics)))

    # Instantiate LightningModule
    lightning_module: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        metrics=metrics,
    )

    # Instantiate all modules relative to the Trainer
    callbacks: List[Callback] = list(hydra.utils.instantiate(cfg.callbacks).values())
    logger: TensorBoardLogger = hydra.utils.instantiate(cfg.logging.logger)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    return trainer, lightning_module, datamodule


def setup_environment():
    """
    Setup environment for training
    """

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
    # Setup environment
    setup_environment()

    # Instantiate three main modules
    trainer, lightning_module, datamodule = instantiate()

    # Train the model ⚡
    trainer.fit(lightning_module, datamodule=datamodule)

    # Test the model
    trainer.test(ckpt_path="best")
