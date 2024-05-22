import torch
from torchmetrics import Metric


class BinaryEmbeddingDistance(Metric):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("score", default=torch.Tensor())
        self.add_state("label", default=torch.Tensor())
        self.add_state("same_mean", default=torch.tensor(0.0))
        self.add_state("same_std", default=torch.tensor(0.0))
        self.add_state("diff_mean", default=torch.tensor(0.0))
        self.add_state("diff_std", default=torch.tensor(0.0))

    def update(self, outputs: dict) -> None:
        """Update state with predictions and targets.

        Args:
        """
        if self.score.shape[0] == 0:
            self.score = outputs["euclidean_distance"]
            self.label = outputs["label"]
        else:
            self.score = torch.cat((self.score, outputs["euclidean_distance"]), dim=0)
            self.label = torch.cat((self.label, outputs["label"]), dim=0)

    def compute(self) -> dict:
        """Computes binary embedding distance."""
        same_distance = self.score[self.label == 1]
        diff_distance = self.score[self.label == 0]

        self.same_mean = torch.mean(same_distance)
        self.diff_mean = torch.mean(diff_distance)
        self.same_std = torch.std(same_distance)
        self.diff_std = torch.std(diff_distance)

        metric_output = {
            "same_speaker_distance_mean": self.same_mean,
            "same_speaker_distance_std": self.same_std,
            "diff_speaker_distance_mean": self.diff_mean,
            "diff_speaker_distance_std": self.diff_std,
        }

        return metric_output
