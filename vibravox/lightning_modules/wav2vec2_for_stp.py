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
            pad_token_id=35,  #self.trainer.datamodule.tokenizer.pad_token_id,
            bos_token_id=36,  #self.trainer.datamodule.tokenizer.bos_token_id,
            eos_token_id=37,  #self.trainer.datamodule.tokenizer.eos_token_id,
            vocab_size=36,  #len(self.trainer.datamodule.tokenizer),
        )

        # pad_token_id = self.tokenizer.pad_token_id,
        # bos_token_id = self.tokenizer.bos_token_id,
        # eos_token_id = self.tokenizer.eos_token_id,
        # vocab_size = len(self.tokenizer),

        self.wav2vec2_for_ctc.freeze_feature_encoder()

        # for param in self.wav2vec2_for_ctc.feature_extractor.parameters():
        #     param.requires_grad = False
        # for param in self.wav2vec2_for_ctc.feature_projection.parameters():
        #     param.requires_grad = False
        # for param in self.wav2vec2_for_ctc.encoder.parameters():
        #     param.requires_grad = False

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

        # Get tensors
        speech = batch["audio"]
        target_ids = batch["phonemes_ids"]

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)

        self.log("training/ctc_loss", forward_result.loss)

        return forward_result.loss

    def validation_step(self, batch):

        # Get tensors
        speech = batch["audio"]
        target_ids = batch["phonemes_ids"]

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)

        return forward_result

    def test_step(self, batch, batch_idx):
        pass

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:

        predicted_ids = torch.argmax(outputs.logits, dim=2)
        target_ids = batch["phonemes_ids"]

        target_ids[
            target_ids == -100
        ] = self.trainer.datamodule.processor.tokenizer.pad_token_id

        predicted_phonemes = self.trainer.datamodule.processor.tokenizer.decode(torch.flatten(predicted_ids))
        target_phonemes = self.trainer.datamodule.processor.tokenizer.decode(torch.flatten(target_ids))

        self.log_dict(
            dictionary=self.metrics(predicted_phonemes, target_phonemes),
            sync_dist=True,
            prog_bar=True,
        )

        self.logger.experiment.add_text(tag="validation/predicted_phonemes",
                                        text_string=str(predicted_phonemes),
                                        global_step=self.trainer.global_step + batch_idx)
        self.logger.experiment.add_text(tag="validation/target_phonemes",
                                        text_string=str(target_phonemes),
                                        global_step=self.trainer.global_step + batch_idx)
        self.log("validation/ctc_loss", outputs.loss)

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            List[torch.optimizer.Optimizer]

        """

        return self.optimizer
