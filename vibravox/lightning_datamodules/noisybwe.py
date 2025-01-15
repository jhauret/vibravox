import torch
import re
from typing import Dict, List, Union
from itertools import cycle
from lightning import LightningDataModule
from datasets import Audio, load_dataset, Dataset, DatasetDict, concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from vibravox.utils import mix_speech_and_noise
from vibravox.utils import set_audio_duration
from vibravox.torch_modules.dsp.data_augmentation import WaveformDataAugmentation
from vibravox.datasets.speech_noise import SpeechNoiseDataset

class NoisyBWELightningDataModule(LightningDataModule):
    
    def __init__(
        self,
        sample_rate: int = 16000,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        subset: str = "speech_clean",
        sensor: str = "headset_microphone",
        collate_strategy: str = "constant_length-2500-ms",
        data_augmentation: torch.nn.Module = None,
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs,
    ):
        """
        LightningDataModule for data-augmented Bandwidth Extension (BWE)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name (str, optional): Dataset name.
                Must be one of "Cnam-LMSSC/vibravox" or "Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp".
                Defaults to "Cnam-LMSSC/vibravox".
            subset (str, optional): Subset. Defaults to "speech_clean"
            sensor (str, optional): Sensor. Defaults to "headset_microphone"
            collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - "pad": Pad the audio signals to the length of the longest signal in the batch.
                - "constant_length-XXX-ms": Cut or pad the audio signals to XXXms.
            Defaults to "constant_length-3000-ms".
            data_augmentation (nn.Module, optional): Data augmentation module. Defaults to None.
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        assert dataset_name in ["Cnam-LMSSC/vibravox", "Cnam-LMSSC/vibravox2", "Cnam-LMSSC/vibravox-test", "Cnam-LMSSC/vibravox_enhanced_by_EBEN"], \
            f"dataset_name {dataset_name} not supported."
        self.dataset_name = dataset_name
        self.subset = subset
        self.subset_speechless_noisy = "speechless_noisy" # for distribution change 
        self.subset_speech_noisy = "speech_noisy" # for test set
        self.sensor = sensor

        assert collate_strategy == "pad" or re.match(r"constant_length-\d+-ms", collate_strategy), \
            "collate_strategy must be 'pad' or match the pattern 'constant_length-XXX-ms'"

        self.collate_strategy = collate_strategy

        if data_augmentation is None:
            data_augmentation = WaveformDataAugmentation(sample_rate)
        assert isinstance(data_augmentation, WaveformDataAugmentation), "data_augmentation must be a WaveformDataAugmentation"

        self.data_augmentation = data_augmentation

        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: str = None):
        """
        Set up the datasets.

        Args:
            stage (str): Pipeline stage among ['fit', 'validate', 'test', 'predict']. Defaults to None.

        Notes:
            This function runs on every accelerator in distributed mode.
            That is why it is necessary to define attributes here rather than in __init__.
        """
        # Load datasets
        speechclean = load_dataset(
            self.dataset_name, self.subset, streaming=self.streaming
        )
        speechless_noisy = load_dataset(
            self.dataset_name, self.subset_speechless_noisy, streaming=self.streaming
        )
        speech_noisy = load_dataset(
            self.dataset_name, self.subset_speech_noisy, streaming=self.streaming
        )

        # Define datasets with their respective suffixes
        # datasets = {
        #     "speechclean": {
        #         "dataset": dataset_dict_speechclean,
        #         "suffix": None
        #     },
        #     "speechless_noisy": {
        #         "dataset": dataset_dict_speechless_noisy,
        #         "suffix": "speechless_noisy"
        #     },
        #     "speech_noisy": {
        #         "dataset": dataset_dict_speech_noisy,
        #         "suffix": None
        #     }
        # }
        
        # Process each dataset: rename columns, select columns, cast and format
        
        speechclean = speechclean.rename_column("audio.headset_microphone", "audio_airborne")
        speechclean = speechclean.rename_column(f"audio.{self.sensor}", "audio_body_conducted")
        speechclean = speechclean.select_columns(["audio_airborne", "audio_body_conducted"])
        speechclean = speechclean.cast_column(
            "audio_airborne", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        speechclean = speechclean.cast_column(
            "audio_body_conducted", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        speechclean = speechclean.with_format("torch")
        
        speechless_noisy = speechless_noisy.rename_column(f"audio.{self.sensor}", "audio_body_conducted_speechless_noisy")
        speechless_noisy = speechless_noisy.select_columns(["audio_body_conducted_speechless_noisy"])
        speechless_noisy = speechless_noisy.cast_column(
            "audio_body_conducted_speechless_noisy", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        speechless_noisy = speechless_noisy.with_format("torch")
            
        speech_noisy = speech_noisy.rename_column(f"audio.{self.sensor}", "audio_body_conducted_speech_noisy")
        speech_noisy = speech_noisy.select_columns(["audio_body_conducted_speech_noisy"])
        speech_noisy = speech_noisy.cast_column(
            "audio_body_conducted_speech_noisy", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        speech_noisy = speech_noisy.with_format("torch")
        
        

        # Process each dataset: rename columns, select columns, cast and format
        # for key, value in datasets.items():
        #     dataset = value["dataset"]
        #     suffix = value["suffix"]
        #     dataset = self._rename_column(dataset, suffix)
        #     datasets[key]["dataset"] = dataset
            
        # datasets=self._add_noise_columns(datasets)
        
        # for key, value in datasets.items():
        #     dataset = value["dataset"]
        #     suffix = value["suffix"]
        #     dataset = self._cast_column(dataset, suffix)
        #     datasets[key]["dataset"] = dataset

        # # Augment datasets
        # datasets["speechclean"]["dataset"] = self.augment_dataset(datasets["speechclean"]["dataset"])

        # Assign datasets based on the stage
        if stage in ["fit", None]:
            
            speech_train = speechclean["train"]
            speech_validation = speechclean["validation"]
            noise_train = speechless_noisy["train"]
            noise_validation = speechless_noisy["validation"]
            
            self.train_dataset = SpeechNoiseDataset(speech_train, noise_train)
            self.val_dataset = SpeechNoiseDataset(speech_validation, noise_validation)
            
        if stage in ["test", None]:
            
            # Concatenate speech_noisy splits
            speech_noisy_real = concatenate_datasets([speech_noisy["train"], speech_noisy["validation"], speech_noisy["test"]])
            
            self.test_dataset = speech_noisy_real

    # def _rename_column(self, dataset: Dataset, suffix: Union[str, None]) -> Dataset:
    #     """
    #     Helper function to rename, select, cast, and format audio columns in a dataset.

    #     Args:
    #         dataset (Dataset): The dataset to process.
    #         suffix (str, optional): Suffix to append to the audio column names. Defaults to None.

    #     Returns:
    #         Dataset: The processed dataset.
    #     """
    #     # Define new column names
    #     audio_airborne = "audio_airborne" + (f"_{suffix}" if suffix else "")
    #     audio_body_conducted = "audio_body_conducted" + (f"_{suffix}" if suffix else "")

    #     # Rename columns
    #     dataset = dataset.rename_column("audio.headset_microphone", audio_airborne)
    #     dataset = dataset.rename_column(f"audio.{self.sensor}", audio_body_conducted)

    #     # Select relevant columns
    #     dataset = dataset.select_columns([audio_airborne, audio_body_conducted])

    #     return dataset
    
    # def _add_noise_columns(self, datasets: Dict[str, Dict[str, Dataset]]) -> Dict[str, Dict[str, Dataset]]:
    #     """_summary_

    #     Args:
    #         datasets (Dict[str, Dict[str, Dataset]]): _description_

    #     Returns:
    #         Dict[str, Dict[str, Dataset]]: _description_
    #     """
    #     # Define column names
    #     audio_airborne_speechless_noisy = "audio_airborne_speechless_noisy"
    #     audio_body_conducted_speechless_noisy = "audio_body_conducted_speechless_noisy"
        
    #     # Add columns
    #     for i in ["train", "validation"]:
    #         datasets["speechclean"]["dataset"][i] = datasets["speechclean"]["dataset"][i].add_column(audio_airborne_speechless_noisy, datasets["speechless_noisy"]["dataset"][i][audio_airborne_speechless_noisy])
    #         datasets["speechclean"]["dataset"][i] = datasets["speechclean"]["dataset"][i].add_column(audio_body_conducted_speechless_noisy, datasets["speechless_noisy"]["dataset"][i][audio_body_conducted_speechless_noisy])
        
    #     return datasets
    
    # def _cast_column(self, dataset: Dataset, suffix: Union[str, None]) -> Dataset:
    #     """_summary_

    #     Args:
    #         dataset (Dataset): _description_
    #         suffix (Union[str, None]): _description_

    #     Returns:
    #         Dataset: _description_
    #     """
    #     # Define new column names
    #     audio_airborne = "audio_airborne" + (f"_{suffix}" if suffix else "")
    #     audio_body_conducted = "audio_body_conducted" + (f"_{suffix}" if suffix else "")

    #     # Cast audio columns
    #     dataset = dataset.cast_column(
    #         audio_airborne, Audio(sampling_rate=self.sample_rate, mono=False)
    #     )
    #     dataset = dataset.cast_column(
    #         audio_body_conducted, Audio(sampling_rate=self.sample_rate, mono=False)
    #     )

    #     # Set format to torch
    #     dataset = dataset.with_format("torch")

    #     return dataset

    # def augment_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    #     """
    #     Augment the speech dataset with noise by mixing audio samples.

    #     Args:
    #         dataset (Union[Dataset, DatasetDict]): The dataset to augment the data.

    #     Returns:
    #         Union[Dataset, DatasetDict]: The augmented dataset with mixed audio.
    #     """
        
    #     _dataset = dataset
        
    #     def map_fn(audio_key: str, noise_key: str):
    #         def _map(batch):
    #             list1 = batch[audio_key]
    #             list2 = batch[noise_key]
    #             mixed_audio = []
    #             noise_cycle = cycle(list2)
    #             # Filter None values out of the lists
    #             # list1 = list(filter(None, list1))
    #             # noise_cycle = cycle(filter(None, list2))
    #             for audio1 in list1:
    #                 audio2 = next(noise_cycle)
    #                 sample1 = audio1["array"]
    #                 sample2 = audio2["array"]
    #                 mixed, _ = mix_speech_and_noise(sample1, sample2)
            
    #                 audio_mixed = {'path': audio1['path'], 'array': mixed, 'sampling_rate': audio1['sampling_rate']}
                    
    #                 mixed_audio.append(audio_mixed)
    #             return {audio_key: mixed_audio}
    #         return _map

    #     # Augment 'audio_airborne'
    #     for i in ["train", "validation"]:
    #         _dataset[i] = _dataset[i].map(
    #             map_fn("audio_airborne", "audio_airborne_speechless_noisy"),
    #             remove_columns=["audio_airborne_speechless_noisy"],
    #             batched=True,
    #         )
    #         _dataset[i] = _dataset[i].map(
    #             map_fn("audio_body_conducted", "audio_body_conducted_speechless_noisy"),
    #             remove_columns=["audio_body_conducted_speechless_noisy"],
    #             batched=True,
    #         )

    #     return _dataset

    def train_dataloader(self):
        """
        Train dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(batch, deterministic=False, collate_strategy=self.collate_strategy)
        )

    def val_dataloader(self):
        """
        Validation dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(batch, deterministic=True, collate_strategy=self.collate_strategy),
        )

    def test_dataloader(self):
        """
        Test dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(batch, deterministic=True, collate_strategy="pad"),
        )

    def data_collator(self, batch: List[Dict[str, Audio]], deterministic: bool, collate_strategy: str) -> Dict[str, torch.Tensor]:
        """
        Custom data collator function to dynamically pad the data.

        Args:
            - batch (List[Dict[str, Audio]]): Dict from the dataset with the keys 'audio_body_conducted' and 'audio_airborne':
                - 'audio_body_conducted': (torch.Tensor of dimension (sample_rate * duration)),
                - 'audio_airborne': (torch.Tensor of dimension (sample_rate * duration))
            - deterministic (bool): If True, always select the same part of the signal.
            - collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - 'pad': Pad the audio signals to the length of the longest signal in the batch,
                - 'constant_length-XXX-ms': Cut or pad the audio signals to XXXms
        
        Returns:
            Dict[str, torch.Tensor] A dictionary containing collated data with keys:
                - 'audio_body_conducted': (torch.Tensor of dimension (batch_size, 1, sample_rate * duration)),
                - 'audio_airborne': (torch.Tensor of dimension (batch_size, 1, sample_rate * duration))
        """

        body_conducted_batch = [item["audio_body_conducted"]["array"] for item in batch]
        air_conducted_batch = [item["audio_airborne"]["array"] for item in batch]

        if collate_strategy == "pad":

            body_conducted_padded_batch = pad_sequence(
                body_conducted_batch, batch_first=True, padding_value=0.0
            ).unsqueeze(1)
            air_conducted_padded_batch = pad_sequence(
                air_conducted_batch, batch_first=True, padding_value=0.0
            ).unsqueeze(1)

        else:
            ms_length = int(self.collate_strategy.split('-')[1])
            samples = int(self.sample_rate * ms_length / 1000)

            body_conducted_padded_batch = []
            air_conducted_padded_batch = []
            for body_conducted, air_conducted in zip(body_conducted_batch, air_conducted_batch):
                body_conducted_padded, air_conducted_padded = set_audio_duration(
                    audio=body_conducted,
                    desired_samples=samples,
                    audio_bis=air_conducted,
                    deterministic=deterministic,
                )
                body_conducted_padded_batch.append(body_conducted_padded.unsqueeze(0))
                air_conducted_padded_batch.append(air_conducted_padded.unsqueeze(0))
            body_conducted_padded_batch = torch.stack(body_conducted_padded_batch, dim=0)
            air_conducted_padded_batch = torch.stack(air_conducted_padded_batch, dim=0)

        # Apply data augmentation
        if deterministic is False:
            with torch.no_grad():
                body_conducted_padded_batch, air_conducted_padded_batch = self.data_augmentation(body_conducted_padded_batch, air_conducted_padded_batch)        

        return {
            "audio_body_conducted": body_conducted_padded_batch,
            "audio_airborne":  air_conducted_padded_batch,
        }