import datetime
import logging
import os
import warnings

import hydra

# import lightning as L
from math import ceil
from omegaconf import DictConfig

# Disable annoying warnings from Huggingface transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Set environment variables for Huggingface cache, for datasets and transformers models
# (should be defined before importing datasets and transformers modules)
dir_huggingface_cache_path: str = "/home/Donnees/Data/Huggingface_cache"
os.environ["HF_HOME"] = dir_huggingface_cache_path
os.environ["HF_DATASETS_CACHE"] = dir_huggingface_cache_path + "/datasets"
os.environ["TRANSFORMERS_CACHE"] = dir_huggingface_cache_path + "/models"

# Set environment variables for full trace of errors
os.environ["HYDRA_FULL_ERROR"] = "1"
dir_path: str = str(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
now: str = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S/")

logger: logging.Logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Args:
        cfg (`DictConfig`): The config file containing parameters.
    """
    datamodule = hydra.utils.instantiate(
        cfg.data_module,
        _recursive_=False,
    )
    nn_module = hydra.utils.instantiate(
        cfg.neural_network,
        nb_steps_per_epoch=ceil(
            datamodule.get_nb_samples_train()
            / (cfg.trainer.gpus * cfg.batch_size * cfg.trainer.accumulate_grad_batches)
        ),
        tokenizer=datamodule.processor.tokenizer,
        _recursive_=False,
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        log_every_n_steps=ceil(
            datamodule.get_nb_samples_train()
            / (
                cfg.trainer.gpus
                * cfg.batch_size
                * cfg.trainer.accumulate_grad_batches
                * cfg.log_n_times_per_epoch
            )
        ),
        callbacks=[
            hydra.utils.instantiate(
                cfg.callbacks.checkpoint_callback_last,
                feature_extractor=datamodule.processor.feature_extractor,
            ),
            hydra.utils.instantiate(
                cfg.callbacks.checkpoint_callback_top_k,
                feature_extractor=datamodule.processor.feature_extractor,
            ),
            hydra.utils.instantiate(cfg.callbacks.lr_logger_callback),
            hydra.utils.instantiate(cfg.callbacks.early_stopping_callback),
            hydra.utils.instantiate(cfg.callbacks.rich_model_summary),
        ],
    )
    logger.info("Trainer trains with model")
    trainer.logger.experiment.add_hparams(
        hparam_dict=hydra.utils.instantiate(cfg.hparam_dict),
        metric_dict={"key": 0.0},
        run_name="./",
    )
    trainer.fit(model=nn_module, datamodule=datamodule)
    trainer.test(
        model=nn_module,
        datamodule=datamodule,
        verbose=True,
    )


if __name__ == "__main__":
    main()
    logger.info("/********************/")
