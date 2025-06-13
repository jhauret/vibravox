from functools import partial
from typing import Any, Dict

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from vibravox.lightning_modules.base_se import BaseSELightningModule


class EBENLightningModule(BaseSELightningModule):
    def __init__(
        self,
        sample_rate: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: partial[torch.optim.Optimizer],
        discriminator_optimizer: partial[torch.optim.Optimizer],
        reconstructive_loss_freq_fn: torch.nn.Module = None,
        reconstructive_loss_time_fn: torch.nn.Module = None,
        feature_matching_loss_fn: torch.nn.Module = None,
        adversarial_loss_fn: torch.nn.Module = None,
        dynamic_loss_balancing: str = None,
        beta_ema: float = 0.9,
        update_discriminator_ratio: float = 1.0,
        description: str = None,
        push_to_hub_after_testing: bool = False,
    ):
        """
        Definition of EBEN and its training pipeline with pytorch lightning paradigm

        Args:
            sample_rate (int): Sample rate of the audio
            generator (torch.nn.Module): Neural network to enhance the speech
            discriminator (torch.nn.Module): Neural networks to discriminate between real and fake audio
            generator_optimizer (partial[torch.optim.Optimizer]): Optimizer for the generator
            discriminator_optimizer (partial[torch.optim.Optimizer]): Optimizer for the discriminator
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
            update_discriminator_ratio (float): Ratio of updates of the discriminator compared to the generator.
                Must be smaller or equal to 1. Default: 1.0
            description (str): Description to log in tensorboard
            push_to_hub_after_testing (bool): If True, the model is pushed to the Hugging Face hub after testing. Defaults to False.
        """

        super().__init__(sample_rate=sample_rate, description=description)

        self.generator: torch.nn.Module = generator
        self.discriminator: torch.nn.Module = discriminator

        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer(params=self.generator.parameters())
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer(
            params=self.discriminator.parameters()
        )

        self.reconstructive_loss_temp_fn: torch.nn.Module = reconstructive_loss_time_fn
        self.reconstructive_loss_freq_fn: torch.nn.Module = reconstructive_loss_freq_fn
        self.feature_matching_loss_fn: torch.nn.Module = feature_matching_loss_fn
        self.adversarial_loss_fn: torch.nn.Module = adversarial_loss_fn

        assert dynamic_loss_balancing in {
            None,
            "simple",
            "ema",
        }, "dynamic_loss_balancing must be in {None, 'simple', 'ema'}"
        self.dynamic_loss_balancing: str = dynamic_loss_balancing
        self.atomic_norms_old = None
        self.beta_ema: float = beta_ema

        assert 0 <= update_discriminator_ratio <= 1, "update_discriminator_ratio must be in [0, 1]"
        self.update_discriminator_ratio: float = update_discriminator_ratio

        self.push_to_hub_after_testing: bool = push_to_hub_after_testing
        self.automatic_optimization: bool = False
        self.first_sample: torch.Tensor = None

    def training_step(self, batch: Dict[str, torch.Tensor]):
        """
        Lightning training step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio_body_conducted", "audio_airborne"
                                                and values of shape (batch_size, channels, samples)
        """

        corrupted_speech = self.generator.cut_to_valid_length(batch["audio_body_conducted"])
        reference_speech = self.generator.cut_to_valid_length(batch["audio_airborne"])
        generator_optimizer, discriminator_optimizer = self.optimizers(use_pl_optimizer=True)

        # Train Generator
        self.toggle_optimizer(generator_optimizer)
        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, "analysis")
        atomic_losses_generator = self.compute_atomic_losses(
            "generator", enhanced_speech, reference_speech, decomposed_enhanced_speech, decomposed_reference_speech
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

        # Train Discriminator
        self.toggle_optimizer(discriminator_optimizer)
        atomic_losses_discriminator = self.compute_atomic_losses(
            "discriminator", enhanced_speech, reference_speech, decomposed_enhanced_speech, decomposed_reference_speech
        )
        if atomic_losses_discriminator and torch.rand(1) < self.update_discriminator_ratio:
            for key, value in atomic_losses_discriminator.items():
                self.log(f"train/discriminator/{key}", value, sync_dist=True)
            backprop_loss_discriminator = (
                atomic_losses_discriminator["real_loss"] + atomic_losses_discriminator["fake_loss"]
            )
            self.log("train/discriminator/backprop_loss", backprop_loss_discriminator, sync_dist=True)
            self.manual_backward(backprop_loss_discriminator)
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()
        self.untoggle_optimizer(discriminator_optimizer)

        return {"corrupted": corrupted_speech, "enhanced": enhanced_speech, "reference": reference_speech}

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """
        return [self.generator_optimizer, self.discriminator_optimizer]

    def common_eval_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, stage: str, dataloader_idx: int
    ) -> STEP_OUTPUT:
        """
        Common evaluation step for validation and test.

        Args:
            batch (Any): Dict with key "audio_body_conducted" ( and optionally "audio_airborne" when reference is available)
                                        and values of shape (batch_size, channels, samples)
            batch_idx (int): Index of the batch
            stage (str): Stage of the evaluation. One of {"validation", "test"}
            dataloader_idx (int): Index of the dataloader

        """
        corrupted_speech = self.generator.cut_to_valid_length(batch["audio_body_conducted"])
        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)

        outputs = {"corrupted": corrupted_speech, "enhanced": enhanced_speech}

        if "audio_airborne" in batch:
            reference_speech = self.generator.cut_to_valid_length(batch["audio_airborne"])
            decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, "analysis")
            outputs["reference"] = reference_speech

            # Log losses
            dl_name = f"/{self.dataloader_names[dataloader_idx]}" if self.dataloader_names else ""
            for net_type in ["generator", "discriminator"]:
                atomic_losses = self.compute_atomic_losses(
                    net_type, enhanced_speech, reference_speech, decomposed_enhanced_speech, decomposed_reference_speech
                )
                for key, value in atomic_losses.items():
                    self.log(f"{stage}/{net_type}/{key}{dl_name}", value, sync_dist=True, add_dataloader_idx=False)
        return outputs

    def on_test_end(self) -> None:
        if self.push_to_hub_after_testing:
            self.generator.push_to_hub(
                f"Cnam-LMSSC/EBEN_{self.trainer.datamodule.sensor}",
                commit_message=f"Upload EBENGenerator after {self.trainer.current_epoch} epochs",
            )

    def compute_atomic_losses(
        self,
        network: str,
        enhanced_speech: torch.Tensor,
        reference_speech: torch.Tensor,
        decomposed_enhanced_speech: torch.Tensor,
        decomposed_reference_speech: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        atomic_losses = dict()
        assert network in {"generator", "discriminator"}
        if network == "generator":
            if self.reconstructive_loss_freq_fn:
                atomic_losses["reconstructive_loss_freq"] = self.reconstructive_loss_freq_fn(
                    enhanced_speech, reference_speech
                )
            if self.reconstructive_loss_temp_fn:
                atomic_losses["reconstructive_loss_temp"] = self.reconstructive_loss_temp_fn(
                    enhanced_speech, reference_speech
                )
            if self.feature_matching_loss_fn or self.adversarial_loss_fn:
                enhanced_embeddings = self.discriminator(bands=decomposed_enhanced_speech, audio=enhanced_speech)
                if self.feature_matching_loss_fn:
                    reference_embeddings = self.discriminator(bands=decomposed_reference_speech, audio=reference_speech)
                    atomic_losses["feature_matching_loss"] = self.feature_matching_loss_fn(
                        enhanced_embeddings, reference_embeddings
                    )
                if self.adversarial_loss_fn:
                    atomic_losses["adv_loss_gen"] = self.adversarial_loss_fn(embeddings=enhanced_embeddings, target=1)
        else:  # discriminator
            if self.adversarial_loss_fn:
                enhanced_embeddings = self.discriminator(
                    bands=decomposed_enhanced_speech.detach(), audio=enhanced_speech.detach()
                )
                reference_embeddings = self.discriminator(bands=decomposed_reference_speech, audio=reference_speech)
                atomic_losses["real_loss"] = self.adversarial_loss_fn(embeddings=reference_embeddings, target=1)
                atomic_losses["fake_loss"] = self.adversarial_loss_fn(embeddings=enhanced_embeddings, target=-1)
        return atomic_losses

    def dynamically_balance_losses(self, atomic_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_adjustment_layer = self.generator.last_conv.weight
        # Compute gradient norms
        atomic_norms = [
            torch.norm(torch.autograd.grad(loss, loss_adjustment_layer, retain_graph=True)[0]).detach()
            for loss in atomic_losses.values()
        ]
        # Update EMA of norms
        if self.atomic_norms_old is None or self.dynamic_loss_balancing == "simple":
            self.atomic_norms_old = atomic_norms
        if self.dynamic_loss_balancing == "ema":
            self.atomic_norms_old = [
                self.beta_ema * old + (1 - self.beta_ema) * new for old, new in zip(self.atomic_norms_old, atomic_norms)
            ]
        # Compute lambdas and apply them
        lambdas = [torch.clamp(1 / (norm + 1e-4), min=0.0, max=1e4) for norm in self.atomic_norms_old]
        for key, lambda_ in zip(atomic_losses.keys(), lambdas):
            atomic_losses[key] *= lambda_
        return atomic_losses
