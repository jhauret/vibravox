import math
from functools import partial
from itertools import chain
from typing import Any, Dict

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from moshi.models import loaders

from vibravox.lightning_modules.base_se import BaseSELightningModule


class RegressiveMimiLightningModule(BaseSELightningModule):
    def __init__(
        self,
        sample_rate: int,
        optimizer: partial[torch.optim.Optimizer],
        loss_feature_fn: torch.nn.Module = None,
        description: str = None,
    ):
        assert sample_rate == 24_000, "sample_rate must be 24_000 Hz for this model"
        super().__init__(sample_rate=sample_rate, description=description)

        weight_path = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.frozen_mimi_model = loaders.get_mimi(weight_path)
        self.trainable_mimi_model = loaders.get_mimi(weight_path)

        self.trainable_mimi_model.train()
        self.trainable_mimi_model.set_num_codebooks(self.trainable_mimi_model.total_codebooks)
        self.frozen_mimi_model.eval()  # Set to eval mode as it's not trained

        self.optimizer = optimizer(
            chain(
                self.trainable_mimi_model.encoder.parameters(),
                self.trainable_mimi_model.encoder_transformer.parameters(),
                self.trainable_mimi_model.downsample.parameters(),
            )
        )
        self.loss_feature_fn = loss_feature_fn

    def training_step(self, batch):

        corrupted_speech = self.pad_to_correct_length(batch["audio_body_conducted"])
        reference_speech = self.pad_to_correct_length(batch["audio_airborne"])
        with torch.no_grad():
            reference_emb = self.frozen_mimi_model.encode_to_latent(reference_speech, quantize=False)
        enhanced_emb = self.trainable_mimi_model.encode_to_latent(corrupted_speech, quantize=False)

        loss_feature = self.loss_feature_fn(enhanced_emb, reference_emb)
        self.log("train/loss_feature", loss_feature, sync_dist=True)

        return loss_feature

    def configure_optimizers(self):
        return self.optimizer

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
        corrupted_speech = self.pad_to_correct_length(batch["audio_body_conducted"])
        enhanced_codes = self.trainable_mimi_model.encode(corrupted_speech)
        enhanced_speech = self.trainable_mimi_model.decode(enhanced_codes)

        outputs = {"corrupted": corrupted_speech, "enhanced": enhanced_speech}

        if "audio_airborne" in batch:
            reference_speech = self.pad_to_correct_length(batch["audio_airborne"])
            outputs["reference"] = reference_speech

            # Log losses
            dl_name = f"/{self.dataloader_names[dataloader_idx]}" if self.dataloader_names else ""
            value = self.loss_feature_fn(
                self.trainable_mimi_model.encode_to_latent(corrupted_speech, quantize=False),
                self.frozen_mimi_model.encode_to_latent(reference_speech, quantize=False),
            )
            self.log(f"{stage}/{dl_name}", value, sync_dist=True, add_dataloader_idx=False)

        return outputs

    @staticmethod
    def pad_to_correct_length(tensor: torch.Tensor):
        """Pad the tensor to a multiple of 1920 for the mimi model."""
        length = tensor.shape[-1]
        pad_to_multiple = 1920
        right_pad = math.ceil(length / pad_to_multiple) * pad_to_multiple - length
        return torch.nn.functional.pad(tensor, (0, right_pad))
