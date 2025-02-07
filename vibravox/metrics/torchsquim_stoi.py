from collections import OrderedDict
from copy import deepcopy
import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE
from torchmetrics import Metric


class TorchsquimSTOI(Metric):
    """Wrapper for the SquimObjective model from Torchaudio to compute STOI.

    This metric calculates the Short-Time Objective Intelligibility (STOI) score,
    which is used to evaluate the intelligibility of speech signals. It leverages
    the SquimObjective model provided by Torchaudio to process prediction tensors.

    Args:
        **kwargs: Additional keyword arguments for the parent `Metric` class.

    Example:
        >>> metric = TorchsquimSTOI()
        >>> preds = torch.randn(1, 16000)
        >>> metric.update(preds)
        >>> stoi_score = metric.compute()
    """
    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize the TorchsquimSTOI metric.

        Sets up the STOI computation model and initializes the state variables.

        Args:
            **kwargs: Additional keyword arguments for the parent `Metric` class.
        """
        super().__init__(**kwargs)

        self.compute_stoi = SQUIM_OBJECTIVE.get_model()
        self.compute_stoi.requires_grad_(False)

        self.add_state("sum_stoi", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # To prevent the states to be recorded in the state_dict
        self._modules_backup = OrderedDict()
        self.register_load_state_dict_post_hook(self.reassign_modules)

    def update(self, preds: torch.Tensor, *args, **kwargs) -> None:
        """Accumulates STOI scores from model predictions.

        Processes the input predictions to compute the STOI score and updates
        the internal state for averaging.

        Args:
            preds (torch.Tensor): Model predictions with shape [..., time].
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Example:
            >>> metric.update(preds)
        """
        stoi_batch, _, _ = self.compute_stoi(preds.view(1, -1))

        self.sum_stoi += stoi_batch.to(self.sum_stoi.device).sum()
        self.total += stoi_batch.numel()

    def compute(self) -> torch.Tensor:
        """Calculates the average STOI score.

        Computes the mean STOI score accumulated over all updates.

        Returns:
            torch.Tensor: The average STOI score.

        Example:
            >>> stoi_score = metric.compute()
        """
        return self.sum_stoi / self.total

    # We do not want to have TorchsquimSTOI in the state dict and trainable parameters
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
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
        # Remove compute_stoi keys if present to avoid unexpected key errors.
        state_dict.pop(prefix + "compute_stoi", None)

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

    def reassign_modules(self, module, incompatible_keys) -> None:
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
