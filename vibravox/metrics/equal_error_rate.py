from typing import Dict, List, Optional, Union, Any

import torch
from torchmetrics import Metric, ROC


class EqualErrorRate(Metric):
    """
    Equal Error Rate Metric

    Given the scores and labels of a binary classifier, this metric computes the Equal Error Rate (EER), defined
    as the point where the False Rejection Rate (FRR) and the False Acceptance Rate (FAR) are equal. The algorithm
    firstly computes the FRR and the FAR for different values of the decision threshold. Then it calculates the optimal
    threshold that minimizes the absolute difference between FRR and FAR. Finally, it computes and returns the
    corresponding EER, FRR and FAR at the optimal threshold.

    By default, the algorithm employs the non-binned approach to calculate the thresholds from the data. It is also
    possible to get a fixed number of threshold linearly spaced from 0 to 1, or to provide a list (or tensor) of
    predefined thresholds.

    The data from one whole epoch is required for the computation of this metric. Therefore, it requires that the
    update() method is called at the end of each batch and the compute() method is called at the end of the epoch.
    Directly calling the forward() method won't give the correct outputs.
    """
    def __init__(
        self,
        score_key: str,
        label_key: str,
        thresholds: Optional[Union[int, List[float], torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the EER metric.

        - Initializes the ROC curve
        - Adds states for the metric's inputs and outputs.

        Args:
            score_key (str): Key for the scores in the model's output dictionary.
            label_key (str): Key for the labels in the model's output dictionary.
            thresholds (Union[int, List[float], torch.Tensor], optional): Threshold parameter for the ROC curve. See
                torchmetrics.ROC for more details. Defaults to None: will use the most accurate non-binned approach to
                dynamically calculate the thresholds.
        """
        super().__init__(**kwargs)

        # Keys for the model scores and labels
        self.score_key = score_key
        self.label_key = label_key

        # Init ROC
        self.roc = ROC(thresholds=thresholds, task="binary")

        # Add states for the metric's inputs and outputs
        self.add_state("score", default=torch.Tensor())
        self.add_state("label", default=torch.Tensor())
        self.add_state("eer", default=torch.tensor(0.0))
        self.add_state("threshold", default=torch.tensor(0.0))
        self.add_state("fr_rate_at_threshold", default=torch.tensor(0.0))
        self.add_state("fa_rate_at_threshold", default=torch.tensor(0.0))

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
        Computes the outputs of the EER metric.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the scalar outputs with keys:
                - "equal_error_rate": Equal Error Rate,
                - "threshold": Binary decision threshold at which the EER is obtained,
                - "false_reject_rate": False rejection rate at threshold,
                - "false_accept_rate": False acceptance rate at threshold,
        """
        # ROC
        fa_rate, ta_rate, thresholds = self.roc(self.score, self.label)
        fr_rate = 1 - ta_rate

        # EER
        eer_threshold_idx = torch.argmin(torch.abs(fa_rate - fr_rate))
        self.eer = (fr_rate[eer_threshold_idx] + fa_rate[eer_threshold_idx]) / 2

        # Optimum threshold
        self.threshold = thresholds[eer_threshold_idx]

        # False reject and acceptance rates at optimum threshold
        self.fr_rate_at_threshold = fr_rate[eer_threshold_idx]
        self.fa_rate_at_threshold = fa_rate[eer_threshold_idx]

        metric_output = {
            "equal_error_rate": self.eer,
            "threshold": self.threshold,
            "false_reject_rate": self.fr_rate_at_threshold,
            "false_accept_rate": self.fa_rate_at_threshold,
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
