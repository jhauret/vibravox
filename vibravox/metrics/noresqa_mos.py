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
        mos_batch = self.compute_mos(preds.squeeze(1), target.squeeze(1)).to(
            self.sum_mos.device
        )
        # Squeeze channel dimension

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
        """Loads the state dictionary, excluding the `compute_stoi` module.

        This method overrides the default `_load_from_state_dict` to prevent
        the `compute_stoi` module from being loaded as a trainable parameter.

        Args:
            state_dict (dict): The state dictionary to load.
            prefix (str): Prefix for state_dict keys.
            local_metadata (dict): Metadata associated with the state_dict.
            strict (bool): Whether to strictly enforce that the keys in state_dict
                match the keys returned by this module's `state_dict` function.
            missing_keys (list): List of missing keys.
            unexpected_keys (list): List of unexpected keys.
            error_msgs (list): List of error messages.
        """
        # Remove compute_mos keys if present to avoid unexpected key errors.
        state_dict.pop(prefix + "compute_mos", None)

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

    def reassign_modules(self, module, incompatible_keys):
        """Reassigns the original modules after state dictionary operations.

        This method restores the original modules from the backup after the state
        dictionary has been loaded to ensure that `compute_stoi` is not included
        as a trainable parameter.

        Args:
            module (torch.nn.Module): The module being loaded.
            incompatible_keys (dict): Dictionary of incompatible keys.
        """
        self._modules = deepcopy(self._modules_backup)
        self._modules_backup = OrderedDict()
