from typing import Dict, Any

import torch
from torchmetrics import Metric


class BinaryEmbeddingDistance(Metric):
    """
    Binary Embedding Distance Metric

    This metric splits the scores of a binary classifier according to the True and False labels and computes the
    respective mean values and standard deviations.

    It is called Binary Embedding Distance, because the classifier usually computes a distance between embeddings to
    get its score, e.g. for a speaker verification task.

    The data from one whole epoch is required for the computation of this metric. Therefore, it requires that the
    update() method is called at the end of each batch and the compute() method is called at the end of the epoch.
    Directly calling the forward() method won't give the correct outputs.
    """
    def __init__(
        self,
        score_key: str,
        label_key: str,
        **kwargs,
    ) -> None:
        """
        Initializes the Binary Embedding Distance metric.

        - Adds states for the metric's inputs and outputs.

        Args:
            score_key (str): Key for the scores in the model's output dictionary.
            label_key (str): Key for the labels in the model's output dictionary.
        """
        super().__init__(**kwargs)

        # Keys for the model scores and labels
        self.score_key = score_key
        self.label_key = label_key

        # Add states for the metric's inputs and outputs
        self.add_state("score", default=torch.Tensor())
        self.add_state("label", default=torch.Tensor())
        self.add_state("same_mean", default=torch.tensor(0.0))
        self.add_state("same_std", default=torch.tensor(0.0))
        self.add_state("diff_mean", default=torch.tensor(0.0))
        self.add_state("diff_std", default=torch.tensor(0.0))

    def update(self, outputs: Dict[str, torch.Tensor]) -> None:
        """
        Updates the input states of the metric from the model scores and labels.

        Args:
            outputs (Dict[str, torch.Tensor]): Dictionary containing model scores and labels as torch.Tensor of
                dimension (batch_size, ).
        """
        if self.score.shape[0] == 0:
            self.score = outputs[self.score_key]
            self.label = outputs[self.label_key]
        else:
            self.score = torch.cat((self.score, outputs[self.score_key]), dim=0)
            self.label = torch.cat((self.label, outputs[self.label_key]), dim=0)

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes Binary Embedding Distance.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the scalar outputs with keys:
                - "same_distance_mean": Mean embedding distance for True labels,
                - "same_distance_std": Standard deviation for True labels,
                - "diff_distance_mean": Mean embedding distance for False labels,
                - "diff_distance_std": Standard deviation for False labels,
        """
        # Separation of scores according to label
        same_distance = self.score[self.label == 1]
        diff_distance = self.score[self.label == 0]

        # Mean and standard deviation for True labels
        self.same_mean = torch.mean(same_distance)
        self.same_std = torch.std(same_distance)

        # Mean and standard deviation for False labels
        self.diff_mean = torch.mean(diff_distance)
        self.diff_std = torch.std(diff_distance)

        metric_output = {
            "same_distance_mean": self.same_mean,
            "same_distance_std": self.same_std,
            "diff_distance_mean": self.diff_mean,
            "diff_distance_std": self.diff_std,
        }

        return metric_output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Raises an error, as only the update() and compute() methods should be called.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            f"The forward() method of this metric is deactivated. "
            f"The update() method should be called at the end of each batch and "
            f"compute() at the end of the epoch."
        )
