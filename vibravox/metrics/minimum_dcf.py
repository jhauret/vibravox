import torch
from torchmetrics import Metric, ROC


class MinimumDetectionCostFunction(Metric):
    def __init__(
        self,
        n_threshold: int,
        target_probability: float = 0.05,
        false_reject_cost: float = 1.0,
        false_accept_cost: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.roc = ROC(thresholds=n_threshold, task="binary")

        self.target_probability = target_probability
        self.false_reject_cost = false_reject_cost
        self.false_accept_cost = false_accept_cost

        self.add_state("score", default=torch.Tensor())
        self.add_state("label", default=list())
        self.add_state("min_dcf", default=torch.tensor(0.0))

    def update(self, outputs: dict) -> None:
        """Update state with predictions and targets.

        Args:
        """
        if self.score.shape[0] == 0:
            self.score = outputs["cosine_similarity"]
            self.label = outputs["label"]
        else:
            self.score = torch.cat((self.score, outputs["cosine_similarity"]), dim=0)
            self.label = torch.cat((self.label, outputs["label"]), dim=0)

    def compute(self) -> dict:
        """Computes Minimum Detection Cost Function."""
        # ROC
        fa_rate, ta_rate, thresholds = self.roc(self.score, self.label)
        fr_rate = 1 - ta_rate

        # Detection Cost Function
        dcf = (
            self.false_reject_cost * self.target_probability * fr_rate
            + self.false_accept_cost * (1 - self.target_probability) * fa_rate
        )
        c_det = torch.min(dcf)
        c_def = min(
            self.false_reject_cost * self.target_probability,
            self.false_accept_cost * (1 - self.target_probability),
        )
        self.min_dcf = c_det / c_def

        metric_output = {
            "minimum_detection_cost_function": self.min_dcf,
        }

        return metric_output
