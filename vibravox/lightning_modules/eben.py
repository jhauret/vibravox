from functools import partial
from typing import Any, Dict, List

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection


class EBENLightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: partial[torch.optim.Optimizer],
        discriminator_optimizer: partial[torch.optim.Optimizer],
        metrics: MetricCollection,
        reconstructive_loss_freq_fn: torch.nn.Module = None,
        reconstructive_loss_time_fn: torch.nn.Module = None,
        feature_matching_loss_fn: torch.nn.Module = None,
        adversarial_loss_fn: torch.nn.Module = None,
        dynamic_loss_balancing: str = None,
        beta_ema: float = 0.9,
        description: str = None,
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
            reconstructive_loss_freq_fn (torch.nn.Module): Function to compute the frequency reconstructive loss (forward call on temporal domain)
            reconstructive_loss_time_fn (torch.nn.Module): Function to compute the temporal reconstructive loss (forward call on temporal domain)
            feature_matching_loss_fn (torch.nn.Module): Function to compute the feature matching loss
            adversarial_loss_fn (torch.nn.Module): Function to compute the adversarial loss
            dynamic_loss_balancing (str): Whether to automatically balance the losses w.r.t. their gradients. One of:
             - None: Do not balance the losses
             - "simple": Balance the losses by dividing them by the gradient norm
             - "ema": Balance the losses by dividing them by the exponential moving average of the gradient norm
             Default: None
            beta_ema (float): Beta parameter for the exponential moving average. Only used if dynamic_loss_balancing="ema". Default: 0.9
            description (str): Description to log in tensorboard
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

        self.reconstructive_loss_temp_fn: torch.nn.Module = reconstructive_loss_time_fn
        self.reconstructive_loss_freq_fn: torch.nn.Module = reconstructive_loss_freq_fn
        self.feature_matching_loss_fn: torch.nn.Module = feature_matching_loss_fn
        self.adversarial_loss_fn: torch.nn.Module = adversarial_loss_fn

        assert dynamic_loss_balancing in {None, "simple", "ema"}, "dynamic_loss_balancing must be in {None, 'simple', 'ema'}"
        self.dynamic_loss_balancing = dynamic_loss_balancing
        self.atomic_norms_old = None  # For dynamic loss balancing
        self.beta_ema = beta_ema

        self.metrics: MetricCollection = metrics
        self.description: str = description

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
        generator_optimizer, discriminator_optimizer = self.optimizers(use_pl_optimizer=True)

        ######################## Train Generator ########################
        self.toggle_optimizer(generator_optimizer)

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, "analysis")

        # Initialize outputs
        outputs = {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            }

        atomic_losses_generator = self.compute_atomic_losses(
            network="generator",
            enhanced_speech=enhanced_speech,
            reference_speech=reference_speech,
            decomposed_enhanced_speech=decomposed_enhanced_speech,
            decomposed_reference_speech=decomposed_reference_speech,
        )

        for key, value in atomic_losses_generator.items():
            self.log(f"train/generator/{key}", value, sync_dist=True)

        if self.dynamic_loss_balancing is not None:
            atomic_losses_generator = self.dynamically_balance_losses(atomic_losses_generator)

        backprop_loss_generator = sum(atomic_losses_generator.values())
        self.log("train/generator/backprop_loss", backprop_loss_generator, sync_dist=True)

        self.manual_backward(backprop_loss_generator)
        generator_optimizer.step()
        generator_optimizer.zero_grad()
        self.untoggle_optimizer(generator_optimizer)

        ######################## Train Discriminator ########################
        self.toggle_optimizer(discriminator_optimizer)
        atomic_losses_discriminator = self.compute_atomic_losses(
            network="discriminator",
            enhanced_speech=enhanced_speech,
            reference_speech=reference_speech,
            decomposed_enhanced_speech=decomposed_enhanced_speech,
            decomposed_reference_speech=decomposed_reference_speech,
        )

        if atomic_losses_discriminator:
            for key, value in atomic_losses_discriminator.items():
                self.log(f"train/discriminator/{key}", value, sync_dist=True)

            backprop_loss_discriminator = atomic_losses_discriminator["real_loss"] + atomic_losses_discriminator["fake_loss"]
            self.log("train/discriminator/backprop_loss", backprop_loss_discriminator, sync_dist=True)

            self.manual_backward(backprop_loss_discriminator)
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()

        self.untoggle_optimizer(discriminator_optimizer)

        return outputs

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
        """
        return self.common_eval_step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
        """
        return self.common_eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return [self.generator_optimizer, self.discriminator_optimizer]

    def on_train_start(self) -> None:
        self.logger.experiment.add_text(tag='description', text_string=self.description)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.common_eval_logging("validation", outputs, batch_idx)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.common_eval_logging("test", outputs, batch_idx)

    def common_eval_step(self, batch, batch_idx, stage):

        assert stage in ["validation", "test"], "stage must be in ['validation', 'test']"

        # Get tensors
        corrupted_speech = self.generator.cut_to_valid_length(batch["audio_body_conducted"])
        reference_speech = self.generator.cut_to_valid_length(batch["audio_airborne"])
        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, "analysis")

        outputs = {
                f"corrupted": corrupted_speech,
                f"enhanced": enhanced_speech,
                f"reference": reference_speech,
            }

        atomic_losses_generator = self.compute_atomic_losses(
            network="generator",
            enhanced_speech=enhanced_speech,
            reference_speech=reference_speech,
            decomposed_enhanced_speech=decomposed_enhanced_speech,
            decomposed_reference_speech=decomposed_reference_speech,
        )

        for key, value in atomic_losses_generator.items():
            self.log(f"{stage}/generator/{key}", value, sync_dist=True)

        atomic_losses_discriminator = self.compute_atomic_losses(
            network="discriminator",
            enhanced_speech=enhanced_speech,
            reference_speech=reference_speech,
            decomposed_enhanced_speech=decomposed_enhanced_speech,
            decomposed_reference_speech=decomposed_reference_speech,
        )

        for key, value in atomic_losses_discriminator.items():
            self.log(f"train/discriminator/{key}", value, sync_dist=True)

        return outputs

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

    def compute_atomic_losses(self,
                              network: str,
                              enhanced_speech: torch.Tensor,
                              reference_speech: torch.Tensor,
                              decomposed_enhanced_speech: torch.Tensor,
                              decomposed_reference_speech: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the atomic losses for the generator or the discriminator

        Args:
            network (str): Network to compute the losses for. One of {"generator", "discriminator"}
            enhanced_speech (torch.Tensor): Enhanced speech of shape (batch_size, 1, samples)
            reference_speech (torch.Tensor): Reference speech of shape (batch_size, 1, samples)
            decomposed_enhanced_speech (torch.Tensor): Decomposed enhanced speech of shape (batch_size, generator.m, samples)
            decomposed_reference_speech (torch.Tensor): Decomposed reference speech (batch_size, generator.m, samples)

        Returns:
            Dict[str, torch.Tensor]: Dictionary with the atomic losses
        """

        atomic_losses = dict()

        assert network in {"generator", "discriminator"}, "network must be in {'generator', 'discriminator'}"

        if network == "generator":
            # Compute losses that are using the discriminator
            if self.reconstructive_loss_freq_fn is not None:
                atomic_losses["reconstructive_loss_freq"] = self.reconstructive_loss_freq_fn(
                    enhanced_speech, reference_speech)
            if self.reconstructive_loss_temp_fn is not None:
                atomic_losses["reconstructive_loss_temp"] = self.reconstructive_loss_temp_fn(
                    enhanced_speech, reference_speech)
            if self.feature_matching_loss_fn is not None or self.adversarial_loss_fn is not None:
                enhanced_embeddings = self.discriminator(
                    bands=decomposed_enhanced_speech, audio=enhanced_speech
                )
                if self.feature_matching_loss_fn is not None:
                    reference_embeddings = self.discriminator(
                        bands=decomposed_reference_speech, audio=reference_speech
                    )
                    atomic_losses["feature_matching_loss"] = self.feature_matching_loss_fn(enhanced_embeddings, reference_embeddings)

                if self.adversarial_loss_fn is not None:
                    atomic_losses["adv_loss_gen"] = self.adversarial_loss_fn(embeddings=enhanced_embeddings, target=1)

        else:
            if self.adversarial_loss_fn is not None:
                enhanced_embeddings = self.discriminator(
                    bands=decomposed_enhanced_speech.detach(), audio=enhanced_speech.detach()
                )
                reference_embeddings = self.discriminator(
                    bands=decomposed_reference_speech, audio=reference_speech
                )

                atomic_losses["real_loss"] = self.adversarial_loss_fn(embeddings=reference_embeddings, target=1)
                atomic_losses["fake_loss"] = self.adversarial_loss_fn(embeddings=enhanced_embeddings, target=-1)

        return atomic_losses

    def compute_lambdas(
        self,
        atomic_losses: Dict[str, torch.Tensor],
        loss_adjustment_layer: torch.Tensor,
        beta: float,
        epsilon: float = 1e-4,
    ):
        """
        Compute the adaptive lambdas to balance the losses

        Args:
            atomic_losses (Dict[torch.Tensor]): Dict of atomic losses
            loss_adjustment_layer (torch.Tensor): Parameters of nn.Module where gradients are computed to adapt lambdas
            beta (float): Beta parameter for the exponential moving average. Only used if ema=True.
            epsilon (float): Small value to avoid division by zero. Default: 1e-4


        Returns:
            List[torch.Tensor]: List of lambdas
        """

        atomic_norms = []

        for atomic_loss in atomic_losses.values():
            atomic_grad = torch.autograd.grad(
                outputs=atomic_loss, inputs=loss_adjustment_layer, retain_graph=True
            )[0]
            atomic_norms.append(torch.norm(atomic_grad).detach())

        if self.atomic_norms_old is None or self.dynamic_loss_balancing == "simple":
            self.atomic_norms_old = atomic_norms

        if self.dynamic_loss_balancing == "ema":
            self.atomic_norms_old = [
                beta * norm_old + (1 - beta) * norm_new
                for norm_old, norm_new in zip(self.atomic_norms_old, atomic_norms)
            ]

        lambdas = [torch.clamp(1 / atomic_norm + epsilon, min=0.0, max=1e4) for atomic_norm in self.atomic_norms_old]

        return lambdas

    def dynamically_balance_losses(self, atomic_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt the weights of the losses according to their gradients

        Args:
            atomic_losses (Dict[str, torch.Tensor]): List of atomic losses
        Returns:
            (Dict[str, torch.Tensor]): List of balanced atomic losses.
        """

        lambdas = self.compute_lambdas(atomic_losses, self.get_loss_adjustment_layer(), self.beta_ema)
        for key, lambda_ in zip(atomic_losses.keys(), lambdas):
            atomic_losses[key] = lambda_ * atomic_losses[key]

        return atomic_losses

    def get_loss_adjustment_layer(self):
        """Get the layer where the gradients are computed to adapt the lambdas

        Returns:
            torch.Tensor: Layer where the gradients are computed
        """
        return self.generator.last_conv.weight
