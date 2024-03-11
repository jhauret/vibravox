from collections import OrderedDict
from copy import deepcopy
import torch
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from torchmetrics import Metric


class NoresqaMOS(Metric):
    """A wrapper for https://pytorch.org/audio/main/generated/torchaudio.models.SquimSubjective.html

    Forward accepts:

    - ``preds``: ``shape [...,time]``
    - ``target``: ``shape [...,time]``

    Args:
        sample_rate: int, sample rate of audio input

    Returns:
        (torch.Tensor) average MOS value over batch
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert sample_rate == 16_000, "NoresqaMOS only supports 16kHz audio"

        self.compute_mos = SQUIM_SUBJECTIVE.get_model()
        self.compute_mos.requires_grad_(False)

        self.add_state("sum_mos", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # To prevent the states to be recorded in the state_dict
        self._modules_backup = OrderedDict()
        self.register_load_state_dict_post_hook(self.reassign_modules)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model
            target (torch.Tensor): non-matching reference ( but can also be the true reference)
        """
        mos_batch = self.compute_mos(preds.squeeze(), target.squeeze()).to(
            self.sum_mos.device
        )

        self.sum_mos += mos_batch.sum()
        self.total += mos_batch.numel()

    def compute(self) -> torch.Tensor:
        """Computes average MOS"""
        return self.sum_mos / self.total

    # We do not want to have NORESQA in the state dict and trainable parameters
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Reloading state_dict that is intialized and static
        state_dict[prefix + "compute_mos"] = self.compute_mos.state_dict(
            prefix=prefix + "compute_mos."
        )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Prevent going into child modules to load as well.
        self._modules_backup = deepcopy(self._modules)
        self._modules = OrderedDict()
        # This is restored in the reassign_modules hook right after (called by load_state_dict)

    def state_dict(
        self,
        destination,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        # We do not want to have NORESQA in the state dict and trainable parameters
        # So we add zeroes in the state_dict
        destination = super().state_dict(destination, prefix, keep_vars)
        for k in list(destination.keys()):
            if "compute_mos" in k:
                destination[k] = torch.Tensor([0.0])
        return destination

    def reassign_modules(self, module, incompatible_keys):
        self._modules = deepcopy(self._modules_backup)
        self._modules_backup = OrderedDict()
