import math
from functools import partial
from itertools import chain
from pathlib import Path
from moshi.models import loaders
from typing import Any, Dict
import torchmetrics.text
from loguru import logger
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchaudio.transforms import Resample


class RegressiveMimiLightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        optimizer: partial[torch.optim.Optimizer],
        codec: str,
        metrics: MetricCollection = None,
        loss_feature_fn: torch.nn.Module = None,
        description: str = None,
    ):
        """
        Args:
            sample_rate (int): Sample rate of the audio
            optimizer (partial[torch.optim.Optimizer]): Optimizer
            metrics (MetricCollection): Metrics to be computed.
            loss_feature_fn (torch.nn.Module): Function to compute the loss from the features
            description (str): Description to log in tensorboard
        """
        super().__init__()

        assert sample_rate == 24_000, "sample_rate must be 24_000 Hz for this model"

        self.sample_rate: int = sample_rate

        weight_path = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)

        self.frozen_mimi_model = loaders.get_mimi(weight_path)
        self.trainable_mimi_model = loaders.get_mimi(weight_path)

        self.trainable_mimi_model.train()

        # Set the number of codebooks to the total number of codebooks in the model for better quality
        self.trainable_mimi_model.set_num_codebooks(self.model.total_codebooks)

        # Only used to compute the reference features
        self.frozen_mimi_model.train()

        self.optimizer = optimizer(
            chain(
                self.trainable_mimi_model.encoder.parameters(),
                self.trainable_mimi_model.encoder_transformer.parameters(),
                self.trainable_mimi_model.downsample.parameters(),
            )
        )

        self.resampler_to_16k = Resample(orig_freq=24_000, new_freq=16_000)
        self.resampler_to_24k = Resample(orig_freq=16_000, new_freq=24_000)

        self.metrics: MetricCollection = metrics
        self.loss_feature_fn = loss_feature_fn
        self.num_val_runs = 0
        self.cer_metric = torchmetrics.text.CharErrorRate()

    def forward(self, batch) -> Any:
        """
        Forward pass of the model
        """

        corrupted_speech = self.pad_to_correct_length(batch["audio_body_conducted"])
        reference_speech = self.pad_to_correct_length(batch["audio_airborne"])

        with torch.no_grad():
            reference_emb = self.frozen_mimi_model.encode_to_latent(reference_speech, quantize=False)

        enhanced_emb = self.trainable_mimi_model.encode_to_latent(corrupted_speech, quantize=False)

        loss_feature = self.loss_feature_fn(enhanced_emb, reference_emb)

        return loss_feature, enhanced_emb, reference_emb

    def training_step(self, batch):
        """
        Lightning training step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
        """

        loss_feature, _, _ = self(batch)
        self.log("train/loss_feature", loss_feature, sync_dist=True)

        return loss_feature

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
        """
        return self.common_eval_step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning testing step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
        """
        return self.common_eval_step(batch, batch_idx, "test")

    @staticmethod
    def pad_to_correct_length(tensor: torch.Tensor):
        """
        Pad the tensor to the correct length for the model forward pass

        Args:
            tensor (torch.Tensor): The input tensor of shape (batch_size, channels, samples)

        Returns:
            torch.Tensor: The padded tensor

        Note: DAC model requires the input to be a multiple of the hop_length (512)
        """

        length = tensor.shape[-1]
        pad_to_multiple = 1920
        right_pad = math.ceil(length / pad_to_multiple) * pad_to_multiple - length
        tensor = torch.nn.functional.pad(tensor, (0, right_pad))

        return tensor

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return self.optimizer

    def on_fit_start(self) -> None:
        """
        Called at the beginning of the fit loop.

        - Checks the consistency of the DataModule's parameters
        - Logs the description in tensorboard.
        """
        self.check_datamodule_parameter()
        self.logger.experiment.add_text(tag="description", text_string=self.description)

    def on_validation_start(self) -> None:
        """
        Called when the validation loop begins.
        """
        self.num_val_runs += 1

    def on_test_start(self) -> None:
        """
        Called at the beginning of the testing loop.

        - Checks the consistency of the DataModule's parameters
        """
        self.check_datamodule_parameter()

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Called at the end of the validation batch. Logs the metrics and audio in tensorboard.

        Args:
            outputs (STEP_OUTPUT): Output of the validation step
            batch (Any): Batch
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        self.common_eval_logging("validation", outputs, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Called at the end of the test batch. Logs the metrics and audio in tensorboard.

        Args:
            outputs (STEP_OUTPUT): Output of the validation step
            batch (Any): Batch
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        self.common_eval_logging("test", outputs, batch_idx, dataloader_idx)

    def common_eval_step(self, batch, batch_idx, stage):
        """
        Common evaluation step for validation and test.

        Args:
            batch (Any): Batch
            batch_idx (int): Index of the batch
            stage (str): Stage of the evaluation. One of {"validation", "test"}

        """

        assert stage in [
            "validation",
            "test",
        ], "stage must be in ['validation', 'test']"

        loss_feature, _, _ = self(batch)
        self.log(f"{stage}/loss_feature", loss_feature, sync_dist=True)

        corrupted_speech = self.pad_to_correct_length(batch["audio_body_conducted"])
        reference_speech = self.pad_to_correct_length(batch["audio_airborne"])

        enhanced_codes = self.trainable_mimi_model.encode(corrupted_speech)
        enhanced_speech = self.trainable_mimi_model.decode(enhanced_codes)

        outputs = {
            f"corrupted": self.resampler_to_16k(corrupted_speech),
            f"enhanced": self.resampler_to_16k(enhanced_speech),
            f"reference": self.resampler_to_16k(reference_speech),
        }

        return outputs

    def common_eval_logging(self, stage, outputs, batch_idx, dataloader_idx=0):
        """
        Common evaluation logging for validation and test.

        Args:
            stage (str): Stage of the evaluation. One of {"validation", "test"}
            outputs (STEP_OUTPUT): Output of the validation step
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        assert stage in [
            "validation",
            "test",
        ], "stage must be in ['validation', 'test']"
        assert "corrupted" in outputs, "corrupted must be in outputs"
        assert "enhanced" in outputs, "enhanced must be in outputs"
        assert "reference" in outputs, "reference must be in outputs"

        # Log metrics
        metrics_to_log = self.metrics(outputs["enhanced"], outputs["reference"])
        metrics_to_log = {f"{stage}/{k}": v for k, v in metrics_to_log.items()}
        self.log_dict(
            dictionary=metrics_to_log,
            sync_dist=True,
            prog_bar=True,
        )

        # Log audio
        if (batch_idx < 15 and self.logger and self.num_val_runs > 1) or stage == "test":
            self.log_audio(
                audio_tensor=outputs["enhanced"],
                tag=f"{stage}_{dataloader_idx}_{batch_idx}/enhanced",
                global_step=self.num_val_runs,
            )
            if self.num_val_runs == 2 or stage == "test":  # 2 because first one is a sanity check in lightning
                self.log_audio(
                    audio_tensor=outputs["reference"],
                    tag=f"{stage}_{dataloader_idx}_{batch_idx}/reference",
                    global_step=self.num_val_runs,
                )
                self.log_audio(
                    audio_tensor=outputs["corrupted"],
                    tag=f"{stage}_{dataloader_idx}_{batch_idx}/corrupted",
                    global_step=self.num_val_runs,
                )

    def log_audio(
        self,
        audio_tensor: torch.Tensor,
        tag: str,
        global_step: int,
    ):
        """
        Log the first audio of the batch of every speech_dict values to tensorboard.

        Args:
            audio_tensor (torch.Tensor): Audio tensor of shape (batch_size, channels, samples) at 48kHz
            tag (str): Tag to identify the audio
            global_step (int): Global step for the audio
        """

        audio_tensor = audio_tensor.detach().cpu()[0, 0, :]
        self.logger.experiment.add_audio(
            tag=tag,
            snd_tensor=audio_tensor,
            global_step=global_step,
            sample_rate=16_000,
        )

    def check_datamodule_parameter(self) -> None:
        """
        List of assertions checking that the parameters of the LightningDatamodule correspond to the LightningModule.

        (Can only be called in stages where the trainer's LightningDataModule is available, e.g. in on_fit_start hook.)

        - Checks the LightningDataModule sample_rate.
        """
        # Check sample rate
        assert self.trainer.datamodule.sample_rate == self.sample_rate, (
            f"sample_rate is not consistent. "
            f"{self.sample_rate} is specified for the LightningModule and "
            f"{self.trainer.datamodule.sample_rate} is provided by the LightningDataModule"
        )
