from typing import Dict, List, Union

import torch

from torch.nn.functional import normalize

from lightning import LightningModule

from torchmetrics import MetricCollection
from torchmetrics.functional import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
)

from huggingface_hub import hf_hub_download


class ECAPA2LightningModule(LightningModule):
    def __init__(
        self,
        metrics: MetricCollection,
    ):
        """
        Initializes the ECAPA2 model with Pytorch Lightning paradigm.

        The model is loaded as a JIT RecursiveScriptModule. Therefore, it can only be employed for testing.

        Args:
            metrics (MetricCollection): collection of metrics to compute
        """
        super().__init__()

        # ECAPA2 model only accepts 16 kHz sample rate and batch_size of 1
        self.sample_rate = 16_000
        self.batch_size = 1

        # Load model from HuggingFace
        self.model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
        self.ecapa2 = torch.jit.load(self.model_file, map_location=self.device)
        self.ecapa2.half()  # optional, but results in faster inference

        # Metrics
        self.metrics = metrics

    def training_step(self, batch):
        """
        Lightning training step: only defined for the trainer

        Returns:
            Nothing
        """

        pass

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step: only defined for the trainer

        Returns:
            Nothing
        """
        pass

    def test_step(
        self,
        batch: Dict[str, Dict[str, Union[torch.Tensor, List[str], List[int]]]],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Lightning testing step

        Computes the normalized embeddings of the ECAPA2 model for both input sensors.

        Args:
            batch (Dict[str, Dict]): Dictionary with keys "sensor_a" and "sensor_b", whose values are dictionaries
                with keys:
                - "audio" (torch.Tensor of dimension (batch_size, 1, sample_rate * duration)),
                - "speaker_id" (List[str]),
                - "sentence_id" (List[torch.Tensor of int]),
                - "gender" (List[str]),
                - "sensor" (List[str])
            batch_idx (int): Index of the batch

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing both sensors' normalized embeddings with keys:
                - "embedding_a" (torch.Tensor of dimension (batch_size, 192)),
                - "embedding_b" (torch.Tensor of dimension (batch_size, 192)),
        """
        # Speaker embeddings for audio A and B
        embedding_a = self.ecapa2(batch["sensor_a"]["audio"])
        embedding_b = self.ecapa2(batch["sensor_b"]["audio"])

        # Normalize the embeddings
        embedding_a = normalize(embedding_a, dim=1)
        embedding_b = normalize(embedding_b, dim=1)

        # Output Dict
        outputs = {
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
        }

        return outputs

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            Nothing

        """

        pass

    def on_test_start(self) -> None:
        """
        Called at the beginning of the testing loop.

        - Checks the LightningDataModule parameters.
        """
        # Check DataModule parameters
        self.check_datamodule_parameter()

    def on_test_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, Union[torch.Tensor, List[str], List[int]]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Called at the end of the test batch.

        - Computes the cosine similarity between the embeddings.
        - Computes the Euclidean distance between the embeddings.
        - Find the test label: is speaker A the same as speaker B?
        - Sends the outputs to the metrics to prepare final computation.

        Args:
            outputs (Dict[str, torch.Tensor]): Dictionary containing both sensors' normalized embeddings with keys:
                - "embedding_a" (torch.Tensor of dimension (batch_size, 192)),
                - "embedding_b" (torch.Tensor of dimension (batch_size, 192)),
            batch (Dict[str, Dict]): Dictionary with keys "sensor_a" and "sensor_b", whose values are dictionaries
                with keys:
                - "audio" (torch.Tensor of dimension (batch_size, 1, sample_rate * duration)),
                - "speaker_id" (List[str]),
                - "sentence_id" (List[torch.Tensor of int]),
                - "gender" (List[str]),
                - "sensor" (List[str])
            batch_idx (int): Index of the batch
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        # Cosine Similarity between the embeddings
        outputs["cosine_similarity"] = pairwise_cosine_similarity(
            outputs["embedding_a"], outputs["embedding_b"]
        ).squeeze(-1)

        # Normalized Euclidean Distance between the embeddings
        outputs["euclidean_distance"] = pairwise_euclidean_distance(
            outputs["embedding_a"], outputs["embedding_b"]
        ).squeeze(-1)

        # True Label: Speaker A == Speaker B ?
        label = [
            int(spk_a == spk_b)
            for spk_a, spk_b in zip(
                batch["sensor_a"]["speaker_id"],
                batch["sensor_b"]["speaker_id"],
            )
        ]
        outputs["label"] = torch.Tensor(label).int().to(self.device)

        # Update the metrics
        self.metrics.update(outputs)

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch.

        - Triggers the computation of the metrics.
        - Logs the metrics to the logger (preference: csv for extracting results as there is no training curves to log).
        """
        # Get the metrics as a dict
        metrics_to_log = self.metrics.compute()

        # Log in the logger
        self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)

    def check_datamodule_parameter(self) -> None:
        """
        List of assertions checking that the parameters of the LightningDatamodule correspond to the LightningModule.

        (Can only be called in stages where the trainer's LightningDataModule is available, e.g. in on_test_start hook.)

        - Checks the LightningDataModule sample_rate.
        - Checks the LightningDataModule batch_size.
        """
        # Check sample rate
        assert self.trainer.datamodule.sample_rate == self.sample_rate, (
            f"sample_rate is not consistent. "
            f"{self.sample_rate} is specified for the LightningModule and "
            f"{self.trainer.datamodule.sample_rate} is provided by the LightningDataModule"
        )

        # Check batch size
        assert self.trainer.datamodule.batch_size == self.batch_size, (
            f"batch_size is not consistent. "
            f"ECAPA2 model only accepts 1 and "
            f"{self.trainer.datamodule.batch_size} is provided by the LightningDataModule"
        )
