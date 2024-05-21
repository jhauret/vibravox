import torch
from torch.nn import CosineSimilarity
from lightning import LightningModule
from torchmetrics import MetricCollection

from huggingface_hub import hf_hub_download


class ECAPA2LightningModule(LightningModule):
    def __init__(
        self,
        sample_rate: int,
        metrics: MetricCollection,
        description: str = None,
    ):
        """
        Definition of ECAPA2 with pytorch lightning paradigm

        Args:
            description (str): Description to log in tensorboard
        """
        super().__init__()

        self.sample_rate = sample_rate

        self.model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
        self.ecapa2 = torch.jit.load(self.model_file)

        self.cosine_similarity = CosineSimilarity(dim=2, eps=1e-8)

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
            outputs (Dict[torch.Tensor, torch.Tensor, List]): Output Dict with keys "cosine similarity",
                                                              "euclidean_distance" and "label"
        """
        # Speaker embeddings for audio A and B
        embedding_a = self.ecapa2(batch["sensor_a"]["audio"])
        embedding_b = self.ecapa2(batch["sensor_b"]["audio"])

        # Cosine Similarity and Euclidean Distance between the embeddings
        cos_sim = self.cosine_similarity(embedding_a, embedding_b)
        distance = torch.cdist(embedding_a, embedding_b, p=2).squeeze(-1)

        # True Label: Speaker A == Speaker B ?
        label = [
            spk_a == spk_b
            for spk_a, spk_b in zip(
                batch["sensor_a"]["speaker_id"], batch["sensor_a"]["speaker_id"]
            )
        ]

        outputs = {
            "cosine_similarity": cos_sim,
            "euclidean_distance": distance,
            "label": label,
        }

        return outputs

    def configure_optimizers(self):
        """
        Method to configure optimizers and schedulers. Automatically called by Lightning's Trainer.

        Returns:
            Nothing

        """

        pass

    def on_test_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Called at the end of the test batch. Send the outputs to the metrics to prepare final computation.
        """
        self.metrics.update(outputs)

    def on_test_end(self) -> None:
        """
        Called at the end of the test on the whole dataset. Perform the metrics computation.
        """
        self.metrics.compute()