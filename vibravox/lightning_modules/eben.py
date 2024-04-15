from functools import partial
from typing import Any, Dict, List

import auraloss
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from vibravox.torch_modules.losses.feature_loss import (
    FeatureLossForDiscriminatorMelganMultiScales,
)
from vibravox.torch_modules.losses.hinge_loss import (
    HingeLossForDiscriminatorMelganMultiScales,
)


class EBENLightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: partial[torch.optim.Optimizer],
        discriminator_optimizer: partial[torch.optim.Optimizer],
        metrics: MetricCollection,
        learning_strategy: str = "all",
        dynamic_loss_balancing: str = None,
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
            learning_strategy (str): Strategy to train the model. Default is "all". Options are:
             - "all": Train both generator and discriminator with all losses
             - "rec_only": Train only the generator with reconstructive losses
             - "adv_only": Train both generator and discriminator with adversarial and feature matching losses
            dynamic_loss_balancing (str): Whether to automatically balance the losses w.r.t. their gradients. One of:
             - None: Do not balance the losses
             - "simple": Balance the losses by dividing them by the gradient norm
             - "ema": Balance the losses by dividing them by the exponential moving average of the gradient norm
             Default: None
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

        assert learning_strategy in {'all', 'rec_only', 'adv_only'}, "learning_strategy must be in {'all', 'rec_only', 'adv_only'}"
        self.learning_strategy = learning_strategy

        if self.learning_strategy in {'all', 'rec_only'}:
            self.reconstructive_loss_temp_fn = torch.nn.L1Loss()
            self.reconstructive_loss_freq_fn = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=[256, 512, 1024, 2048],
                hop_sizes=[128, 256, 512, 1024],
                win_lengths=[256, 512, 1024, 2048],
                scale="mel",
                n_bins=128,
                sample_rate=self.sample_rate,
                perceptual_weighting=True,
            )
        if self.learning_strategy in {'all', 'adv_only'}:
            self.feature_matching_loss_fn = FeatureLossForDiscriminatorMelganMultiScales()
            self.adversarial_loss_fn = HingeLossForDiscriminatorMelganMultiScales()

        assert dynamic_loss_balancing in {None, "simple", "ema"}, "dynamic_loss_balancing must be in {None, 'simple', 'ema'}"
        self.dynamic_loss_balancing = dynamic_loss_balancing
        self.lambdas_past = None  # For dynamic loss balancing

        self.metrics: MetricCollection = metrics

        self.automatic_optimization = False

    def training_step(self, batch):
        """
        Lightning training step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
        """

        # Get tensors
        corrupted_speech = self.generator.cut_to_valid_length(batch["audio_body_conducted"])
        reference_speech = self.generator.cut_to_valid_length(batch["audio_airborne"])

        # Get optimizers
        generator_optimizer, discriminator_optimizer = self.optimizers(
            use_pl_optimizer=True
        )

        ######################## Train Generator ########################
        self.toggle_optimizer(generator_optimizer)

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, "analysis")

        # Initialize step_output
        step_output = {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            }

        atomic_losses = dict()

        # Compute losses that are using the discriminator
        if self.learning_strategy in {"all", "adv_only"}:
            enhanced_embeddings = self.discriminator(
                bands=decomposed_enhanced_speech, audio=enhanced_speech
            )
            reference_embeddings = self.discriminator(
                bands=decomposed_reference_speech, audio=reference_speech
            )

            atomic_losses["adv_loss_gen"] = self.adversarial_loss_fn(embeddings=enhanced_embeddings, target=1)
            atomic_losses["feature_matching_loss"] = self.feature_matching_loss_fn(enhanced_embeddings, reference_embeddings)

            self.log("train/generator/adv_loss", atomic_losses["adv_loss_gen"])
            self.log("train/generator/feature_matching_loss", atomic_losses["feature_matching_loss"])

        # Compute reconstructive losses
        if self.learning_strategy in {"all", "rec_only"}:

            atomic_losses["reconstructive_loss_temp"] = self.reconstructive_loss_temp_fn(enhanced_speech, reference_speech)
            atomic_losses["reconstructive_loss_freq"] = self.reconstructive_loss_freq_fn(enhanced_speech, reference_speech)

            self.log("train/generator/reconstructive_loss_temp", atomic_losses["reconstructive_loss_temp"])
            self.log("train/generator/reconstructive_loss_freq", atomic_losses["reconstructive_loss_freq"])

        # Dynamically balance losses
        if self.dynamic_loss_balancing is not None:
            atomic_losses = self.dynamically_balance_losses(atomic_losses)

        backprop_loss_gen = sum(atomic_losses.values())
        self.log("train/generator/backprop_loss", backprop_loss_gen)

        self.manual_backward(backprop_loss_gen)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)

        ######################## Train Discriminator ########################
        if self.learning_strategy in {"all", "adv_only"}:
            self.toggle_optimizer(discriminator_optimizer)

            # Compute forwards again is necessary because we haven't retain_graph
            enhanced_embeddings = self.discriminator(
                bands=decomposed_enhanced_speech.detach(), audio=enhanced_speech.detach()
            )
            reference_embeddings = self.discriminator(
                bands=decomposed_reference_speech, audio=reference_speech
            )

            real_loss = self.adversarial_loss_fn(embeddings=reference_embeddings, target=1)
            fake_loss = self.adversarial_loss_fn(embeddings=enhanced_embeddings, target=-1)

            backprop_loss_dis = real_loss + fake_loss

            # Log scalars
            self.log("train/discriminator/real_loss", real_loss)
            self.log("train/discriminator/fake_loss", fake_loss)
            self.log("train/discriminator/backprop_loss", backprop_loss_dis)

            self.manual_backward(backprop_loss_dis)
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()
            self.untoggle_optimizer(discriminator_optimizer)

        return step_output

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
        """
        return self.common_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
        """
        return self.common_eval_step(batch, batch_idx)

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

        self.common_eval_logging("validation", outputs, batch_idx)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.common_eval_logging("test", outputs, batch_idx)

    def common_eval_step(self, batch, batch_idx):

        # Get tensors
        corrupted_speech = self.generator.cut_to_valid_length(batch["audio_body_conducted"])
        reference_speech = self.generator.cut_to_valid_length(batch["audio_airborne"])
        enhanced_speech, _ = self.generator(corrupted_speech)

        step_output = {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            }

        return step_output

    def common_eval_logging(self, stage, outputs, batch_idx):

        assert stage in ["validation", "test"], "stage must be in ['validation', 'test']"
        assert "corrupted" in outputs, "corrupted must be in outputs"
        assert "enhanced" in outputs, "enhanced must be in outputs"
        assert "reference" in outputs, "reference must be in outputs"

        # Log metrics
        metrics_to_log = self.metrics(
            outputs["enhanced"], outputs["reference"]
        )
        metrics_to_log = {f"{stage}/{k}": v for k, v in metrics_to_log.items()}
        self.log_dict(
            dictionary=metrics_to_log,
            sync_dist=True,
            prog_bar=True,
        )

        if self.trainer.global_step % 10 == 0:
            # Log audio
            self.log_audio(
                prefix=f"{stage}/", speech_dict=outputs, batch_idx=batch_idx
            )

    @staticmethod
    def ready_to_log(audio_tensor):
        audio_tensor = audio_tensor.detach().cpu()[0, 0, :]
        return audio_tensor

    def log_audio(
        self, speech_dict: Dict[str, torch.Tensor], prefix: str, batch_idx: int = 0
    ):
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

    def compute_lambdas(
        self,
        atomic_losses: Dict[str, torch.Tensor],
        loss_adjustment_layer: torch.Tensor,
        beta: float = 0.999,
    ):
        """
        Compute the adaptive lambdas to balance the losses

        Args:
            atomic_losses (Dict[torch.Tensor]): List of atomic losses
            loss_adjustment_layer (torch.Tensor): Parameters of nn.Module where gradients are computed to adapt lambdas
            beta (float): Beta parameter for the exponential moving average. Only used if ema=True. Default: 0.99

        Returns:
            List[torch.Tensor]: List of lambdas
        """

        lambdas = []

        for atomic_loss in atomic_losses.values():
            atomic_grad = torch.autograd.grad(
                outputs=atomic_loss, inputs=loss_adjustment_layer, retain_graph=True
            )[0]
            lambda_adaptive = 1 / (torch.norm(atomic_grad) + 1e-4)
            lambdas.append(torch.clamp(lambda_adaptive, 0.0, 1e4).detach())

        if self.dynamic_loss_balancing == "ema":
            if self.lambdas_past is None:
                self.lambdas_past = lambdas
            else:
                lambdas = [
                    beta * lambda_past + (1 - beta) * lambda_
                    for lambda_past, lambda_ in zip(self.lambdas_past, lambdas)
                ]

        return lambdas
