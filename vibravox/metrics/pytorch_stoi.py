import torch
from torch_stoi import NegSTOILoss
from torchmetrics import Metric


class PytorchSTOI(Metric):
    """A wrapper for the pytorch_stoi package https://github.com/mpariente/pytorch_stoi

    Forward accepts:

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        sample_rate: int, sample rate of audio input
        use_vad: bool = True, Whether to use simple VAD
        extended: bool = False, Whether to compute extended version of stoi
        do_resample: bool = True, Whether to resample audio input to sample_rate
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.


    Returns:
        (torch.Tensor) average pytorch_stoi value over batch
    """

    def __init__(
        self,
        sample_rate: int,
        use_vad: bool = True,
        extended: bool = False,
        do_resample: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.compute_stoi = NegSTOILoss(
            sample_rate=sample_rate,
            use_vad=use_vad,
            extended=extended,
            do_resample=do_resample,
        )

        self.add_state("sum_stoi", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model
            target (torch.Tensor): Ground truth values
        """
        stoi_batch = -self.compute_stoi(preds, target).to(self.sum_stoi.device)

        self.sum_stoi += stoi_batch.sum()
        self.total += stoi_batch.numel()

    def compute(self) -> torch.Tensor:
        """Computes average STOI."""
        return self.sum_stoi / self.total
