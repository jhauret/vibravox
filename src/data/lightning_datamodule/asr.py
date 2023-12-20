# import lightning as L
import logging
import os
from typing import Dict, Union
import datasets
import hydra
import pandas
import pytorch_lightning as L
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor

# Set environment variables for Huggingface cache, for datasets and transformers models
# (should be defined before importing datasets and transformers modules)
dir_huggingface_cache_path: str = "/home/Donnees/Data/Huggingface_cache"
os.environ["HF_HOME"] = dir_huggingface_cache_path
os.environ["HF_DATASETS_CACHE"] = f"{dir_huggingface_cache_path}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{dir_huggingface_cache_path}/models"

# Set environment variables for full trace of errors
os.environ["HYDRA_FULL_ERROR"] = "1"
dir_path: str = str(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
data_path: str = "/home/Donnees/Data/asr_vibravox/"

logger: logging.Logger = logging.getLogger(__name__)


class ASRLightningDataModule(L.LightningDataModule):
    """
    Constructs a data_module. Inherits from LightningDataModule.
    The dataset is already prepared. The class just fetches it from the file system.

    Args:
        dataset_configuration (str): The configuration of the data downloaded; expl: 'clean' or 'other'.
        dataset_name (str): The name of the dataset; expl: 'librispeech_asr'.
        language (str): The language used to record the dataset; expl: 'en'.
        min_input_length_in_sec (float): Each audio of the dataset is filtered by its length. 0.0 means that the audios are not filtered.
        max_input_length_in_sec (float): Each audio of the dataset is filtered by its length. -1 means that the audios are not filtered.
        processor_class (str): The class of the processor one wants to instantiate; expl: 'transformers.Wav2Vec2Processor'.
        processor (str): Path for retrieving the processor.
        ckpt_path (str): Path for retrieving the checkpoint.
        number_of_hours_train (int): How many hours there are in the training dataset.
        batch_size (int): Defaults to 1. The batch size.
        num_workers (int): Defaults to 0. How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        pin_memory (bool): Defaults to True. If True, the dataloader will copy Tensors into CUDA pinned memory before returning them.
        sampling_rate (int): Defaults to 16000 Hz. The sampling rate of the audio from the considered dataset.
        is_test (bool): Defaults to False. Determines whether we are training + validating or only testing.
        is_continue_training (bool): Defaults to False. Determines if the training continues from ckpt.
        task_type (str): Defaults to 'phoneme'. Whether to do Speech-to-Text or Speech-to-Phoneme.
    """

    def __init__(
        self,
        dataset_configuration: str,
        dataset_name: str,
        language: str,
        min_input_length_in_sec: float,
        max_input_length_in_sec: float,
        processor_class: str,
        processor: str,
        ckpt_path: str,
        number_of_hours_train: int,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        sampling_rate: int = 16000,
        is_test: bool = False,
        is_continue_training: bool = False,
        task_type: str = "phoneme",
    ):
        super(ASRLightningDataModule, self).__init__()
        self.dataset_name: str = dataset_name
        self.dataset_configuration: str = dataset_configuration
        self.language: str = language
        self.max_input_length_in_sec: float = float(max_input_length_in_sec)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.sampling_rate: int = sampling_rate
        self.number_of_hours_train: int = number_of_hours_train
        self.task_type = task_type

        if is_test:
            self.processor: Wav2Vec2Processor = hydra.utils.get_method(
                f"{processor_class}.from_pretrained"
            )(f"{dir_path}/models/{ckpt_path}")
        elif is_continue_training:
            self.processor: Wav2Vec2Processor = hydra.utils.get_method(
                f"{processor_class}.from_pretrained"
            )(f"{dir_path}/models/{ckpt_path}")
        else:
            self.processor: Wav2Vec2Processor = hydra.utils.get_method(
                f"{processor_class}.from_pretrained"
            )(f"{dir_path}{processor}")

    def prepare_data(self) -> None:
        None

    def setup(self, stage) -> None:
        # Load the datasets and split by training and validation splits
        partial_path = f"{data_path}{self.dataset_name}_filtered/{self.language}/{str(int(self.max_input_length_in_sec)) if not self.max_input_length_in_sec == -1 else 'no_filter'}/{self.dataset_configuration}"

        lfd_train = datasets.load_from_disk(dataset_path=f"{partial_path}/train")
        lfd_eval = datasets.load_from_disk(dataset_path=f"{partial_path}/validation")
        lfd_test = datasets.load_from_disk(dataset_path=f"{partial_path}/test")

        logger.info(f"Dataset {self.dataset_name} {self.dataset_configuration}")
        logger.info(
            f"Training on {self.number_of_hours_train if not self.number_of_hours_train == -1 else 'all'} hours of {self.dataset_configuration} samples: {len(lfd_train)}, Validation on samples: {len(lfd_eval)}, Test on samples: {len(lfd_test)}"
        )
        self.nb_samples_train = len(lfd_train)

        self.train_ds: ASRDataset = ASRDataset(
            lfd_train, self.processor, self.task_type
        )
        self.valid_ds: ASRDataset = ASRDataset(lfd_eval, self.processor, self.task_type)
        self.test_ds: ASRDataset = ASRDataset(lfd_test, self.processor, self.task_type)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: ctc_data_collator(batch, self.processor),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: ctc_data_collator(batch, self.processor),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=lambda batch: ctc_data_collator(batch, self.processor),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_nb_samples_train(self) -> int:
        """
        The function returns the length of self.train_ds.

        Returns:
            int: nb_samples_train
        """
        self.setup("fit")
        return self.nb_samples_train


class ASRDataset(Dataset):
    """
    Constructs a ASRDataset. Inherits from torch.utils.data.Dataset
    Overrides __get_item__ and __len__ abstract methods.

    Args:
        ds (`Union[Dataset, DatasetDict]`): The dataset.
        processor (`Wav2Vec2Processor`): The processor needed to process the audio.
        task_type (str): Defaults to 'phoneme'. Whether to do Speech-to-Text or Speech-to-Phoneme.
    """

    def __init__(
        self,
        ds: Union[Dataset, DatasetDict],
        processor: Wav2Vec2Processor,
        task_type: str = "phoneme",
    ):
        super(Dataset, self).__init__()
        self.ds: Union[Dataset, DatasetDict] = ds
        self.processor: Wav2Vec2Processor = processor
        self.task_type = task_type
        self.target_sampling_rate: int = self.processor.feature_extractor.sampling_rate

    def __getitem__(self, idx: int) -> Dict:
        audio = self.processor(
            self.ds[idx]["audio"]["array"], sampling_rate=self.target_sampling_rate
        ).input_values[0]

        with self.processor.as_target_processor():
            labels = self.processor(self.ds[idx][self.task_type]).input_ids
        return {"audio": audio, "label": labels}

    def __len__(self) -> int:
        return len(self.ds)


def ctc_data_collator(
    batch: Union[pandas.DataFrame, Dataset], processor: Wav2Vec2Processor
) -> Dict[torch.tensor, torch.tensor]:
    """
    Custom data collator function to dynamically pad the data.

    Args:
        batch (`Union[pandas.DataFrame, Dataset]`): The dataset.
        processor (`Wav2Vec2Processor`): The processor needed to process the audio.
    Returns:
        `Union[pandas.DataFrame, Dataset]`: batch with padding
    """
    input_features = [{"input_values": sample["audio"]} for sample in batch]
    label_features = [{"input_ids": sample["label"]} for sample in batch]
    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

    # Replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )
    batch["labels"] = labels

    return batch
