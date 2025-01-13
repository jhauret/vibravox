from functools import partial
from typing import Any, Dict

import torch
import transformers
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from transformers import Wav2Vec2Processor


class Wav2Vec2ForSTPLightningModule(LightningModule):
    def __init__(
        self,
        wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC,
        optimizer: partial[torch.optim.Optimizer],
        metrics: MetricCollection,
        push_to_hub_after_testing: bool = False,
    ):
        """
        Definition of Wav2Vec2ForSTP and its training pipeline with pytorch lightning paradigm

        Args:
            wav2vec2_for_ctc (torch.nn.Module): Neural network to enhance the speech
            optimizer (partial[torch.optim.Optimizer]): Optimizer
            metrics (MetricCollection): Metrics to be computed.
            push_to_hub_after_testing (bool): If True, the model is pushed to the Hugging Face hub after testing. Defaults to False.
        """
        super().__init__()

        self.sample_rate: int = 16_000
        self.wav2vec2_for_ctc: transformers.Wav2Vec2ForCTC = wav2vec2_for_ctc(
            pad_token_id=35,  # Corresponds to `self.trainer.datamodule.tokenizer.pad_token_id`
            vocab_size=38,  # Corresponds to `len(self.trainer.datamodule.tokenizer)`
        )

        self.optimizer: torch.optim.Optimizer = optimizer(params=self.wav2vec2_for_ctc.parameters())

        self.metrics: MetricCollection = metrics
        self.push_to_hub_after_testing: bool = push_to_hub_after_testing

    def training_step(self, batch: Dict[str, torch.Tensor]):
        """
        Lightning training step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
        """

        return self.common_step(batch)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx, dataloader_idx=0):
        """
        Lightning validation step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        return self.common_step(batch)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx, dataloader_idx=0):
        """
        Lightning test step

        Args:
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx (int): Index of the batch
            dataloader_idx (int): Index of the dataloader
        """

        return self.common_step(batch)

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
        """
        self.check_datamodule_parameter()

    def on_test_start(self) -> None:
        """
        Called at the beginning of the testing loop.

        - Checks the consistency of the DataModule's parameters
        """
        self.check_datamodule_parameter()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """
        Method automatically called when the train batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the training_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """

        self.common_logging("train", outputs, batch, batch_idx)

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Method automatically called when the validation batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the validation_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """

        self.common_logging("validation", outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Method automatically called when the test batch ends.

        Args:
            outputs (STEP_OUTPUT): Output of the test_step method
            batch (Any): Batch
            batch_idx (int): Index of the batch
        """
        self.common_logging("test", outputs, batch, batch_idx, dataloader_idx)

    def on_test_end(self) -> None:
        """
        Method to be called when the test ends.
        """
        if self.push_to_hub_after_testing:
            self.wav2vec2_for_ctc.push_to_hub(
                f"Cnam-LMSSC/phonemizer_{self.trainer.datamodule.sensor}",
                commit_message=f"Upload Wav2Vec2ForCTC after {self.trainer.current_epoch} epochs",
            )
            processor = Wav2Vec2Processor(
                feature_extractor=self.trainer.datamodule.feature_extractor, tokenizer=self.trainer.datamodule.tokenizer
            )
            processor.push_to_hub(
                f"Cnam-LMSSC/phonemizer_{self.trainer.datamodule.sensor}",
                commit_message=f"Upload Wav2Vec2Processor after {self.trainer.current_epoch} epochs",
            )

    def common_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Common step for training, validation and test steps.

        Args:
             batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"

        Returns:
            Dict[str, torch.Tensor]: Dict with keys "loss", "logits"
        """

        # Get tensors
        speech = batch["audio"]
        target_ids = batch["phonemes_ids"]

        # Forward pass
        forward_result = self.wav2vec2_for_ctc(input_values=speech, labels=target_ids)

        return forward_result

    def common_logging(
        self, stage: str, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Common logging for training, validation and test steps.

        Args:
            stage(str): Stage of the training
            outputs(STEP_OUTPUT): Output of the {train,validation,test}_step method
            batch (Dict[str, torch.Tensor]): Dict with keys "audio", "phonemes_ids", "phonemes_str"
            batch_idx(int): Index of the batch
            dataloader_idx(int): Index of the dataloader
        """

        # Log loss
        self.log(f"{stage}_{dataloader_idx}/ctc_loss", outputs["loss"], sync_dist=True)

        # Log metrics
        predicted_phonemes = self.get_phonemes_from_logits(outputs["logits"])
        target_phonemes = batch["phonemes_str"]
        metrics_to_log = self.metrics(predicted_phonemes, target_phonemes)
        metrics_to_log = {f"{stage}_{dataloader_idx}/{k}": v for k, v in metrics_to_log.items()}

        self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)

        # Log text
        text_to_log = f"OUT: {predicted_phonemes[0]}" + "  \n" + f"GT:{target_phonemes[0]} "
        self.logger.experiment.add_text(
            tag=f"{stage}_{dataloader_idx}/predicted_vs_target__phonemes",
            text_string=text_to_log,
            global_step=self.trainer.global_step + batch_idx,
        )

    def get_phonemes_from_logits(self, model_logits):
        """
        Get phonemes from model logits

        Args:
            model_logits(torch.Tensor): Model logits

        Returns:
            List[str]: List of predicted phonemes
        """

        # Get predicted phonemes
        predicted_ids = torch.argmax(model_logits, dim=2)
        predicted_phonemes = [
            self.trainer.datamodule.tokenizer.decode(predicted_ids[i, :]) for i in range(predicted_ids.shape[0])
        ]

        return predicted_phonemes

    def check_datamodule_parameter(self) -> None:
        """
        List of assertions checking that the parameters of the LightningDatamodule correspond to the LightningModule.

        (Can only be called in stages where the trainer's LightningDataModule is available, e.g. in on_fit_start hook.)

        - Checks the LightningDataModule sample_rate.
        - Checks tokenizer's pad_token_id.
        - Checks the length of the tokenizer.
        """
        # Check sample rate
        assert self.trainer.datamodule.sample_rate == self.sample_rate, (
            f"sample_rate is not consistent. "
            f"{self.sample_rate} is specified for the LightningModule and "
            f"{self.trainer.datamodule.sample_rate} is provided by the LightningDataModule"
        )

        # Check tokenizer's pad_token_id
        assert self.trainer.datamodule.tokenizer.pad_token_id == 35, "Pad token id must be 35"

        # Check length of tokenizer
        assert len(self.trainer.datamodule.tokenizer) == 38, "Vocab size must be 38"
