from functools import partial
from typing import Any

import torch
import transformers
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
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
        self.wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC = wav2vec2_for_ctc(
            pad_token_id=35,  # Corresponds to `self.trainer.datamodule.tokenizer.pad_token_id`
            vocab_size=38,  # Corresponds to `len(self.trainer.datamodule.tokenizer)`
        )

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

        return self.common_step(batch)

    def validation_step(self, batch):
        """
        Lightning validation step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of speech and phonemes
        """

        return self.common_step(batch)

    def test_step(self, batch, batch_idx):
        """
        Lightning test step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Tuple of speech and phonemes
        """

        return self.common_step(batch)

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return self.optimizer

    def on_train_start(self) -> None:
        """
        Method to be called when the train starts.
        """

        assert (
            self.trainer.datamodule.tokenizer.pad_token_id == 35
        ), "Pad token id must be 35"
        assert len(self.trainer.datamodule.tokenizer) == 38, "Vocab size must be 38"

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:

        self.common_logging("train", outputs, batch, batch_idx)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.common_logging("validation", outputs, batch, batch_idx)

    def on_test_batch_end(
            self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        self.common_logging("test", outputs, batch, batch_idx)

    def common_step(self, batch):

        # Get tensors
        speech = batch["audio"]
        target_ids = batch["phonemes_ids"]

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)

        return forward_result

    def common_logging(self, stage, outputs, batch, batch_idx):
        # Log loss
        self.log(f"{stage}/ctc_loss", outputs["loss"], sync_dist=True)

        # Log metrics
        predicted_phonemes = self.get_phonemes_from_logits(outputs["logits"])
        target_phonemes = batch["phonemes_str"]
        metrics_to_log = self.metrics(predicted_phonemes, target_phonemes)
        metrics_to_log = {f"{stage}/{k}": v for k, v in metrics_to_log.items()}

        self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)

        # Log text
        text_to_log = f"OUT: {predicted_phonemes[0]} "+"\n"+f"GT:{target_phonemes[0]} "
        self.logger.experiment.add_text(
            tag=f"{stage}/predicted_vs_target__phonemes",
            text_string=text_to_log,
            global_step=self.trainer.global_step + batch_idx,
        )

    def get_phonemes_from_logits(self, model_logits):

        # Get predicted phonemes
        predicted_ids = torch.argmax(model_logits, dim=2)
        predicted_phonemes = [
            self.trainer.datamodule.tokenizer.decode(predicted_ids[i, :])
            for i in range(predicted_ids.shape[0])
        ]

        return predicted_phonemes
