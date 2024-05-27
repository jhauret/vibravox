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
        description: str = None,
    ):
        """
        Initialize the ECAPA2 model with the specified parameters

        Args:
            metrics (MetricCollection): collection of metrics to compute
            description (str): description to log in tensorboard
        """
        super().__init__()

        # ECAPA2 model only accepts 16 kHz sample rate and batch_size of 1
        self.sample_rate = 16_000
        self.batch_size = 1

        self.model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
        self.ecapa2 = torch.jit.load(self.model_file, map_location=self.device)
        self.ecapa2.half()  # optional, but results in faster inference

        self.metrics = metrics

        self.description: str = description

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

    def test_step(self, batch, batch_idx):
        """
        Lightning validation step

        Args:
            batch (Dict[Dict, Dict]): Dict with keys "sensor_a" and "sensor_b", whose values are Dict with keys
                                        "audio" and "speaker_id".
            batch_idx (int): Index of the batch

        Returns:
            outputs (Dict[torch.Tensor, torch.Tensor]): Output Dict with keys "cosine similarity",
                                                              "euclidean_distance" and "label"
        """
        # Speaker embeddings for audio A and B
        embedding_a = self.ecapa2(batch["sensor_a"]["audio"])
        embedding_b = self.ecapa2(batch["sensor_b"]["audio"])

        # Normalize the embeddings
        embedding_a = normalize(embedding_a, dim=1)
        embedding_b = normalize(embedding_b, dim=1)

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
        Called at the beginning of the testing loop. Logs the description in tensorboard.
        """
        # Check sample rate
        assert self.trainer.datamodule.sample_rate == self.sample_rate, (
            f"sample_rate is not consistent. "
            f"ECAPA2 model only accepts 16_000 and "
            f"{self.trainer.datamodule.sample_rate} is provided by the LightningDataModule"
        )

        # Check batch size
        assert self.trainer.datamodule.batch_size == self.batch_size, (
            f"batch_size is not consistent. "
            f"ECAPA2 model only accepts 1 and "
            f"{self.trainer.datamodule.batch_size} is provided by the LightningDataModule"
        )

        self.logger.experiment.add_text(tag="description", text_string=self.description)

    def on_test_batch_end(
        self, outputs: dict, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Called at the end of the test batch. Send the outputs to the metrics to prepare final computation.
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
        metrics_to_log = self.metrics.compute()
        self.log_dict(dictionary=metrics_to_log, sync_dist=True, prog_bar=True)
