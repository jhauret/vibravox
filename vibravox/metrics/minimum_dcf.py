from typing import Dict, List, Optional, Union, Any

import torch
from torchmetrics import Metric, ROC


class MinimumDetectionCostFunction(Metric):
    """
    Normalized Minimum Detection Cost Function Metric

    The Detection Cost Function (DCF) is defined as a weighted sum of false rejection and false acceptance
    probabilities (cf. NIST 2018 Speaker Recognition Evaluation Plan `[1]`_).
    The algorithm computes the DCF from the scores and labels of a binary classifier for different values of the
    decision threshold between 0 and 1. The parameters are the weights of the sum, namely the cost of a false rejection
    and the cost of a false acceptance, and the a priori probability of the target label.
    This metric normalizes the minimum value of the DCF by dividing it by a default cost, defined as the best cost that
    could be obtained by either always accepting of rejecting the classifier's score `[1]`_.
    This leads to the normalized minimum of the detection cost function.

    By default, the algorithm employs the non-binned approach to calculate the thresholds from the data. It is also
    possible to get a fixed number of threshold linearly spaced from 0 to 1, or to provide a list (or tensor) of
    predefined thresholds.

    The data from one whole epoch is required for the computation of this metric. Therefore, it requires that the
    update() method is called at the end of each batch and the compute() method is called at the end of the epoch.
    Directly calling the forward() method won't give the correct outputs.

    .. _[1]:
        https://www.nist.gov/system/files/documents/2018/08/17/sre18_eval_plan_2018-05-31_v6.pdf
    """
    def __init__(
        self,
        score_key: str,
        label_key: str,
        thresholds: Optional[Union[int, List[float], torch.Tensor]] = None,
        target_probability: float = 0.05,
        false_reject_cost: float = 1.0,
        false_accept_cost: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Initializes the normalized MinDCF metric.

        - Initializes the ROC curve
        - Adds states for the metric's inputs and outputs.

        Args:
            score_key (str): Key for the scores in the model's output dictionary.
            label_key (str): Key for the labels in the model's output dictionary.
            thresholds (Union[int, List[float], torch.Tensor], optional): Threshold parameter for the ROC curve. See
                torchmetrics.ROC for more details. Defaults to None: will use the most accurate non-binned approach to
                dynamically calculate the thresholds.
            target_probability (float, optional): A priori probability of the target label. Defaults to 0.05.
            false_reject_cost (float, optional): Cost of a missed detection. Defaults to 1.
            false_accept_cost (float, optional): Cost of a spurious detection. Defaults to 1.
        """
        super().__init__(**kwargs)

        # Keys for the model scores and labels
        self.score_key = score_key
        self.label_key = label_key

        # Init ROC
        self.roc = ROC(thresholds=thresholds, task="binary")

        # Metric parameters
        self.target_probability = target_probability
        self.false_reject_cost = false_reject_cost
        self.false_accept_cost = false_accept_cost

        # Add states for the metric's inputs and outputs
        self.add_state("score", default=torch.Tensor())
        self.add_state("label", default=list())
        self.add_state("min_dcf", default=torch.tensor(0.0))

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
        Computes the Normalized Minimum Detection Cost Function.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the scalar output with keys:
                - "minimum_detection_cost_function": Normalized minimum detection cost.
        """
        # ROC
        fa_rate, ta_rate, thresholds = self.roc(self.score, self.label)
        fr_rate = 1 - ta_rate

        # Detection Cost Function
        dcf = (
                self.false_reject_cost * self.target_probability * fr_rate
                + self.false_accept_cost * (1 - self.target_probability) * fa_rate
        )

        # Find the minimum of the cost function
        c_det = torch.min(dcf)

        # Normalization with the default cost
        c_def = min(
            self.false_reject_cost * self.target_probability,
            self.false_accept_cost * (1 - self.target_probability),
        )
        self.min_dcf = c_det / c_def

        metric_output = {
            "minimum_detection_cost_function": self.min_dcf,
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
