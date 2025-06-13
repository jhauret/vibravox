from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from torchmetrics import MetricCollection
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, ShortTimeObjectiveIntelligibility

from vibravox.metrics.noresqa_mos import NoresqaMOS
from vibravox.metrics.torchsquim_stoi import TorchsquimSTOI


class BaseSELightningModule(LightningModule, ABC):
    """
    A base LightningModule for speech enhancement tasks to reduce boilerplate code.

    It handles common logic for:
    - Storing common parameters (`sample_rate`, `description`).
    - Standard validation and test steps that delegate to a `common_eval_step`.
    - Standard `on_..._batch_end` hooks that delegate to `common_eval_logging`.
    - Common lifecycle hooks (`on_fit_start`, `on_validation_start`, `on_test_start`).
    - Helper methods for logging audio and checking datamodule parameters.

    Subclasses must implement:
    - `common_eval_step`: The core logic for a validation/test step.
    - `common_eval_logging`: The logic for logging metrics and audio after an eval step.
    """

    def __init__(
        self,
        sample_rate: int,
        description: str,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metrics"])

        self.sample_rate = sample_rate
        self.metrics = MetricCollection(
            dict(
                noresqa_mos=NoresqaMOS(sample_rate=16000),
                torchmetrics_si_sdr=ScaleInvariantSignalDistortionRatio(),
                torchmetrics_stoi=ShortTimeObjectiveIntelligibility(fs=16000),
                torchsquim_stoi=TorchsquimSTOI(),
            )
        )

        self.description = description

        self.num_val_runs: int = 0
        self.dataloader_names: List[str] = None

    # --- Abstract Methods (to be implemented by subclasses) ---

    @abstractmethod
    def common_eval_step(self, batch: Any, batch_idx: int, stage: str, dataloader_idx: int) -> STEP_OUTPUT:
        """
        Common evaluation step for validation and test.
        Must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def common_eval_logging(self, stage: str, outputs: STEP_OUTPUT, batch_idx: int, dataloader_idx: int) -> None:
        """
        Common evaluation logging for validation and test.
        Must be implemented by the subclass.
        """
        pass

    # --- Concrete Implementations of Lightning Hooks ---

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.common_eval_step(batch, batch_idx, "validation", dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.common_eval_step(batch, batch_idx, "test", dataloader_idx)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.common_eval_logging("validation", outputs, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.common_eval_logging("test", outputs, batch_idx, dataloader_idx)

    def on_fit_start(self) -> None:
        self.check_datamodule_parameter()
        if self.logger and self.description:
            self.logger.experiment.add_text(tag="description", text_string=self.description)
        if hasattr(self.trainer.datamodule, "val_dataloader") and isinstance(
            self.trainer.datamodule.val_dataloader(), dict
        ):
            self.dataloader_names = list(self.trainer.datamodule.val_dataloader().keys())

    def on_validation_start(self) -> None:
        self.num_val_runs += 1

    def on_test_start(self) -> None:
        self.check_datamodule_parameter()
        if hasattr(self.trainer.datamodule, "test_dataloader") and isinstance(
            self.trainer.datamodule.test_dataloader(), dict
        ):
            self.dataloader_names = list(self.trainer.datamodule.test_dataloader().keys())

    # --- Common Helper Methods ---

    def log_audio(self, audio_tensor: torch.Tensor, tag: str, global_step: int, sample_rate: int):
        """
        Log the first audio of the batch of every speech_dict values to tensorboard.

        Args:
            audio_tensor (torch.Tensor): Audio tensor of shape (batch_size, channels, samples) at 48kHz
            tag (str): Tag to identify the audio
            global_step (int): Global step for the audio
        """
        if not self.logger:
            return

        audio_tensor = audio_tensor.detach().cpu()[0, 0, :]
        self.logger.experiment.add_audio(
            tag=tag,
            snd_tensor=audio_tensor,
            global_step=global_step,
            sample_rate=sample_rate,
        )

    def check_datamodule_parameter(self) -> None:
        """Asserts the sample rate of the datamodule matches the model's."""
        if not hasattr(self.trainer, "datamodule") or self.trainer.datamodule is None:
            return

        assert self.trainer.datamodule.sample_rate == self.sample_rate, (
            f"sample_rate is not consistent. "
            f"{self.sample_rate} is specified for the LightningModule and "
            f"{self.trainer.datamodule.sample_rate} is provided by the LightningDataModule"
        )
