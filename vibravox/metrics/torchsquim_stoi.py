from collections import OrderedDict
from copy import deepcopy
import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
from torchmetrics import Metric


class TorchsquimSTOI(Metric):
    """A wrapper for https://pytorch.org/audio/main/generated/torchaudio.models.SquimObjective.html

    Forward accepts:

    - ``preds``: ``shape [...,time]``

    Returns:
        (torch.Tensor) 
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.compute_stoi = SQUIM_OBJECTIVE.get_model()
        self.compute_stoi.requires_grad_(False)

        self.add_state("sum_stoi", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # To prevent the states to be recorded in the state_dict
        self._modules_backup = OrderedDict()
        self.register_load_state_dict_post_hook(self.reassign_modules)

    def update(self, preds: torch.Tensor, **kwargs) -> None:
        """Update state with predictions and ignore kwargs.

        Args:
            preds (torch.Tensor): Predictions from model
        """
        stoi_batch, _, _ = self.compute_stoi(preds.view(1, -1)).to(
            self.sum_stoi.device
        )

        self.sum_mos += stoi_batch.sum()
        self.total += stoi_batch.numel()

    def compute(self) -> torch.Tensor:
        """Computes average STOI"""
        return self.sum_stoi / self.total

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
        state_dict[prefix + "compute_stoi"] = self.compute_stoi.state_dict(
            prefix=prefix + "compute_stoi."
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
            if "compute_stoi" in k:
                destination[k] = torch.Tensor([0.0])
        return destination

    def reassign_modules(self, module, incompatible_keys):
        self._modules = deepcopy(self._modules_backup)
        self._modules_backup = OrderedDict()
