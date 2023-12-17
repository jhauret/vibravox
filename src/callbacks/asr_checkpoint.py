import logging
import shutil
from datetime import timedelta
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.types import _METRIC, _PATH
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

logger: logging.Logger = logging.getLogger(__name__)


class ASRCheckpoint(ModelCheckpoint):
    """
    Inherits ModelCheckpoint class and its attributes.

    Args:
        feature_extractor (`SequenceFeatureExtractor`): Feature extractor to save.
    """

    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        every_n_val_epochs: Optional[int] = None,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            every_n_val_epochs,
        )
        self.feature_extractor = feature_extractor
        self.path = dirpath + filename
        self.last_path = ""

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]
    ) -> None:
        """
        Inherits _save_last_checkpoint.

        Args:
            trainer (`pl.Trainer`): pl.Trainer instance.
            monitor_candidates (Dict[str, _METRIC]): Contains valuable attributes of ModelCheckpoint.
        Returns:
            NoneType: None
        """
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )
        trainer.save_checkpoint(filepath, self.save_weights_only)

        """
        OVERRIDDEN
        """
        path = self.format_checkpoint_name(
            metrics=monitor_candidates, filename=self.path
        )
        self._save_model(trainer=trainer, filepath=path, del_filepath=self.last_path)

        if self.last_model_path and self.last_model_path != filepath:
            trainer.training_type_plugin.remove_checkpoint(self.last_model_path)

        self.last_model_path = filepath
        self.last_path = path

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, _METRIC],
    ) -> None:
        """
        Inherits _update_best_and_save.

        Args:
            current (torch.Tensor):
            trainer (`pl.Trainer`): pl.Trainer instance.
            monitor_candidates (Dict[str, _METRIC]): Contains valuable attributes of ModelCheckpoint.
        Returns:
            NoneType: None
        """
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device
            )

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(
                self.best_k_models, key=self.best_k_models.get
            )
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates.get("epoch")
            step = monitor_candidates.get("step")
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:0.5f}"
                f' (best {self.best_model_score:0.5f}), saving model to "{filepath}" as top {k}'
            )
        trainer.save_checkpoint(filepath, self.save_weights_only)

        """
        OVERRIDDEN
        """
        self._save_model(trainer=trainer, filepath=filepath, del_filepath=del_filepath)

        if del_filepath is not None and filepath != del_filepath:
            trainer.training_type_plugin.remove_checkpoint(del_filepath)

    @rank_zero_only
    def _save_model(
        self, trainer: "pl.Trainer", filepath: _PATH, del_filepath: _PATH
    ) -> None:
        """

        Args:
            trainer (`pl.Trainer`): pl.Trainer instance.
            filepath (_PATH): Absolute path to checkpoint to save.
            del_filepath (_PATH): Absolute path to checkpoint to save that must be replaced by newest/best checkpoint.
        Returns:
            NoneType: None
        """
        logger.info("Saving best/last model")
        trainer.lightning_module.model.save_pretrained(
            save_directory=filepath.replace(".ckpt", "")
        )
        trainer.lightning_module.tokenizer.save_pretrained(
            save_directory=filepath.replace(".ckpt", "")
        )
        self.feature_extractor.save_pretrained(
            save_directory=filepath.replace(".ckpt", "")
        )
        if del_filepath is not None and filepath != del_filepath:
            try:
                logger.info("Deleting and replacing best/last model")
                shutil.rmtree(del_filepath.replace(".ckpt", ""))
            except:
                None
