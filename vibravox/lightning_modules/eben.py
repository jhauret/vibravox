from functools import partial

import torch
from lightning import LightningModule

from vibravox.torch_modules.feature_loss import FeatureLossForDiscriminatorMelganMultiScales
from vibravox.torch_modules.hinge_loss import HingeLossForDiscriminatorMelganMultiScales


class EBENLightningModule(LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: partial[torch.optim.Optimizer],
        discriminator_optimizer: partial[torch.optim.Optimizer],
    ):
        """
        Definition of EBEN and its training pipeline with pytorch lightning paradigm

        Args:
            generator (torch.nn.Module): Neural network to enhance the speech
            discriminator (torch.nn.Module): Neural networks to discriminate between real and fake audio
            generator_optimizer (partial[torch.optim.Optimizer]): Optimizer for the generator
            discriminator_optimizer (partial[torch.optim.Optimizer]): Optimizer for the discriminator
        """
        super().__init__()

        self.generator: torch.nn.Module = generator
        self.discriminator: torch.nn.Module = discriminator
        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer(params=self.generator.parameters())
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer(
            params=self.discriminator.parameters()
        )
        self.adversarial_loss = FeatureLossForDiscriminatorMelganMultiScales()
        self.feature_matching_loss = HingeLossForDiscriminatorMelganMultiScales()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Lightning training step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of corrupted and reference speech
        """

        # Get speeches
        cut_batch = [self.generator.cut_to_valid_length(speech) for speech in batch]
        corrupted_speech, reference_speech = cut_batch

        # Get optimizers
        generator_optimizer, discriminator_optimizer = self.optimizers(use_pl_optimizer=True)

        ######################## Train Generator ########################
        self.toggle_optimizer(generator_optimizer)

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, "analysis")
        enhanced_embeddings = self.discriminator(bands=decomposed_enhanced_speech[:, 1:, :], audio=enhanced_speech)
        reference_embeddings = self.discriminator(bands=decomposed_reference_speech[:, 1:, :], audio=reference_speech)

        # Compute adversarial_loss
        adv_loss_gen = self.adversarial_loss(embeddings=enhanced_embeddings, target=1)

        # Compute feature_matching_loss
        feature_matching_loss = self.feature_matching_loss(enhanced_embeddings, reference_embeddings)

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

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return [self.generator_optimizer, self.discriminator_optimizer]