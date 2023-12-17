import logging
import os
from hydra import compose, initialize

# Disable annoying warnings from Huggingface transformers
import warnings
import hydra

# import lightning as L
from omegaconf import DictConfig

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

logger: logging.Logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """
    Test program.

    Args:
        cfg (`DictConfig`): The config file containing parameters.
    """
    datamodule = hydra.utils.instantiate(
        cfg.data_module,
        _recursive_=False,
    )
    nn_module_ckpt = hydra.utils.instantiate(
        cfg.neural_network,
        tokenizer=datamodule.processor.tokenizer,
        _recursive_=False,
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer,
    )
    trainer.logger.experiment.add_hparams(
        hparam_dict=hydra.utils.instantiate(cfg.hparam_dict),
        metric_dict={"key": 0.0},
        run_name="./",
    )
    logger.info("Trainer tests with model")
    trainer.test(
        model=nn_module_ckpt,
        datamodule=datamodule,
        verbose=True,
        ckpt_path=f"{dir_path}/models/{cfg.ckpt}.ckpt",
    )


if __name__ == "__main__":
    with initialize(config_path="../../configs", job_name="test_model"):
        cfg = compose(
            config_name="config",
            overrides=[
                "is_test=True",
                "hydra.run.dir=outputs/test/${now:%Y-%m-%d}/${now:%H-%M-%S}",
                "trainer.gpus=1",
                "trainer.strategy=",
                f"trainer.logger.save_dir={dir_path}",
                "trainer.logger.version=test/${now:%Y-%m-%d}/${now:%H-%M-%S}/logs",
            ],
        )
        main(cfg)
        logger.info("%%% %%% %%% %%% %%%")
