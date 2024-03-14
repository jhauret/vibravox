from functools import partial

import torch
import transformers
from lightning import LightningModule
from torchmetrics import MetricCollection


class Wav2Vec2ForSTPLightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC,
        optimizer: partial[torch.optim.Optimizer],
        metrics: MetricCollection,
    ):
        """
        Definition of EBEN and its training pipeline with pytorch lightning paradigm

        Args:
            sample_rate (int): Sample rate of the audio
            wav2vec2_for_ctc (torch.nn.Module): Neural network to enhance the speech
            optimizer (partial[torch.optim.Optimizer]): Optimizer
            metrics (MetricCollection): Metrics to be computed.
        """
        super().__init__()

        self.sample_rate: int = sample_rate
        self.wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC = wav2vec2_for_ctc
        self.optimizer: torch.optim.Optimizer = optimizer(
            params=self.wav2vec2_for_ctc.parameters()
        )

        self.metrics: MetricCollection = metrics

    def training_step(self, batch):
        """
        Lightning training step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of speech and phonemes
        """

        # Get speeches
        speech, target_ids = batch

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)

        loss = forward_result.loss

        # For logging
        logits = forward_result.logits
        predicted_ids = torch.argmax(logits, dim=2)
        predicted_ids = torch.flatten(predicted_ids)
        reference_ids = torch.flatten(target_ids)

        reference_ids[reference_ids == -100] = self.trainer.datamodule.tokenizer.pad_token_id

        predicted_phonemes = self.trainer.datamodule.tokenizer.decode(predicted_ids)
        target_phonemes = self.trainer.datamodule.tokenizer.decode(reference_ids, group_tokens=False)

        return loss

    def validation_step(self, batch):

        # Get speeches
        speech, target_ids = batch

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)

        loss = forward_result.loss

        # For logging
        logits = forward_result.logits
        predicted_ids = torch.argmax(logits, dim=2)
        predicted_ids = torch.flatten(predicted_ids)
        reference_ids = torch.flatten(target_ids)

        reference_ids[reference_ids == -100] = self.trainer.datamodule.tokenizer.pad_token_id

        predicted_phonemes = self.trainer.datamodule.tokenizer.decode(predicted_ids)
        target_phonemes = self.trainer.datamodule.tokenizer.decode(reference_ids, group_tokens=False)

        self.log_dict(
            dictionary=self.metrics(predicted_phonemes,target_phonemes),
            sync_dist=True,
            prog_bar=True,
        )

        self.logger.experiment.add_text("comparison", str(predicted_phonemes))

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return self.optimizer

