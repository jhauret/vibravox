import torch
from torchmetrics import Metric, ROC


class EqualErrorRate(Metric):
    def __init__(
        self,
        n_threshold: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.roc = ROC(thresholds=n_threshold, task="binary")

        self.add_state("score", default=torch.Tensor())
        self.add_state("label", default=torch.Tensor())
        self.add_state("eer", default=torch.tensor(0.0))
        self.add_state("threshold", default=torch.tensor(0.0))
        self.add_state("fr_rate_at_threshold", default=torch.tensor(0.0))
        self.add_state("fa_rate_at_threshold", default=torch.tensor(0.0))

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
        """Computes Equal Error Rate."""
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
