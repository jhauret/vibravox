from functools import partial
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from vibravox.torch_modules.feature_loss import (
    FeatureLossForDiscriminatorMelganMultiScales,
)
from vibravox.torch_modules.hinge_loss import HingeLossForDiscriminatorMelganMultiScales


class EBENLightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: partial[torch.optim.Optimizer],
        discriminator_optimizer: partial[torch.optim.Optimizer],
        metrics: MetricCollection,
    ):
        """
        Definition of EBEN and its training pipeline with pytorch lightning paradigm

        Args:
            sample_rate (int): Sample rate of the audio
            generator (torch.nn.Module): Neural network to enhance the speech
            discriminator (torch.nn.Module): Neural networks to discriminate between real and fake audio
            generator_optimizer (partial[torch.optim.Optimizer]): Optimizer for the generator
            discriminator_optimizer (partial[torch.optim.Optimizer]): Optimizer for the discriminator
            metrics (MetricCollection): Metrics to be computed.
        """
        super().__init__()

        self.sample_rate: int = sample_rate
        self.generator: torch.nn.Module = generator
        self.discriminator: torch.nn.Module = discriminator
        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer(
            params=self.generator.parameters()
        )
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer(
            params=self.discriminator.parameters()
        )
        self.feature_matching_loss = FeatureLossForDiscriminatorMelganMultiScales()
        self.adversarial_loss = HingeLossForDiscriminatorMelganMultiScales()
        self.metrics: MetricCollection = metrics

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        """
        Lightning training step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of corrupted and reference speech
        """

        # Get speeches
        cut_batch = [self.generator.cut_to_valid_length(speech) for speech in batch]
        corrupted_speech, reference_speech = cut_batch

        # Get optimizers
        generator_optimizer, discriminator_optimizer = self.optimizers(
            use_pl_optimizer=True
        )

        ######################## Train Generator ########################
        self.toggle_optimizer(generator_optimizer)

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(
            reference_speech, "analysis"
        )

        # Initialize step_output
        step_output = {
            # "loss": None, we don't need this because we are using manual optimization
            "audio": {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            },
            "scalars_to_log": dict(),
        }

        # enhanced_embeddings = self.discriminator(bands=decomposed_enhanced_speech[:, 1:, :], audio=enhanced_speech)
        # reference_embeddings = self.discriminator(bands=decomposed_reference_speech[:, 1:, :], audio=reference_speech)
        enhanced_embeddings = self.discriminator(enhanced_speech)
        reference_embeddings = self.discriminator(reference_speech)

        # Compute adversarial_loss
        adv_loss_gen = self.adversarial_loss(embeddings=enhanced_embeddings, target=1)

        # Compute feature_matching_loss
        feature_matching_loss = self.feature_matching_loss(
            enhanced_embeddings, reference_embeddings
        )

        # Compute loss to backprop on
        backprop_loss_gen = adv_loss_gen + feature_matching_loss

        self.manual_backward(backprop_loss_gen)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)

        ######################## Train Discriminator ########################
        self.toggle_optimizer(discriminator_optimizer)

        # Compute forwards again is necessary because we haven't retain_graph
        enhanced_embeddings = self.discriminator(enhanced_speech.detach())
        reference_embeddings = self.discriminator(reference_speech)

        # Compute adversarial_loss
        real_loss = self.adversarial_loss(embeddings=reference_embeddings, target=1)
        fake_loss = self.adversarial_loss(embeddings=enhanced_embeddings, target=-1)

        # Compute and log total loss
        backprop_loss_dis = real_loss + fake_loss

        self.manual_backward(backprop_loss_dis)
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        return step_output

    def validation_step(self, batch, batch_idx):
        cut_batch = [self.generator.cut_to_valid_length(speech) for speech in batch]
        corrupted_speech, reference_speech = cut_batch
        enhanced_speech, _ = self.generator(corrupted_speech)

        step_output = {
            "audio": {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            },
            "scalars_to_log": dict(),
        }

        return step_output

    def test_step(self, batch, batch_idx):
        cut_batch = [self.generator.cut_to_valid_length(speech) for speech in batch]
        corrupted_speech, reference_speech = cut_batch
        enhanced_speech, _ = self.generator(corrupted_speech)

        step_output = {
            "audio": {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            },
            "scalars_to_log": dict(),
        }

        return step_output

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return [self.generator_optimizer, self.discriminator_optimizer]

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        assert "audio" in outputs, "audio key must be in outputs"

        self.log_dict(
            prefix="validation/metrics/",
            dictionary=self.metrics(outputs["audio"]["enhanced"], outputs["audio"]["reference"]),
            sync_dist=True,
            prog_bar=True,
        )

        self.log_audio(prefix="validation/", speech_dict=outputs["audio"], batch_idx=batch_idx)

    def log_dict(self, *args, **kwargs):
        """
        Logging passed dictionary and adding prefix on logging name if defined
        """

        if "dictionary" in kwargs and "prefix" in kwargs:
            dictionary = kwargs["dictionary"]
            prefix = kwargs.pop("prefix")
            kwargs["dictionary"] = {f"{prefix}{k}": v for k, v in dictionary.items()}
            return super().log_dict(*args, **kwargs)
        else:
            return super().log_dict(*args, **kwargs)


    @staticmethod
    def ready_to_log(audio_tensor):
        audio_tensor = audio_tensor.detach().cpu()[0, 0, :]
        return audio_tensor

    def log_audio(self, speech_dict: Dict[str, torch.Tensor], prefix: str, batch_idx: int = 0):
        """
        Log the first audio of the batch of every speech_dict values to tensorboard.

        Args:
            speech_dict (Dict[str, torch.Tensor]): Dictionary of tensors of shape (batch_size, channels, samples)
            prefix (str): Prefix to be added to the name of the audio
            batch_idx (int): Batch index to be added to the global_step as `self.trainer.global_step` only counts optimizer steps
        """

        if self.logger:
            for speech_name, speech_tensor in speech_dict.items():
                speech_tensor = self.ready_to_log(speech_tensor)
                self.logger.experiment.add_audio(
                    tag=f"{prefix}{speech_name}",
                    snd_tensor=speech_tensor,
                    global_step=self.trainer.global_step + batch_idx,
                    sample_rate=self.sample_rate,
                )
