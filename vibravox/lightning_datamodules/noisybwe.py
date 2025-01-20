import torch
import re
from typing import Dict, List
from lightning import LightningDataModule
from datasets import Audio, load_dataset, concatenate_datasets
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
        snr_range: List[float] = [-3.0, 5.0],
        **kwargs,
    ):
        """
        LightningDataModule for Noisy Bandwidth Extension (BWE)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name (str, optional): Dataset name.
                Must be one of "Cnam-LMSSC/vibravox" or "Cnam-LMSSC/vibravox_enhanced_by_EBEN".
                Defaults to "Cnam-LMSSC/vibravox".
            subset (str, optional): Subset. Defaults to "speech_clean"
            sensor (str, optional): Sensor. Defaults to "headset_microphone"
            collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - "pad": Pad the audio signals to the length of the longest signal in the batch.
                - "constant_length-XXX-ms": Cut or pad the audio signals to XXXms.
            Defaults to "constant_length-2500-ms".
            data_augmentation (nn.Module, optional): Data augmentation module. Defaults to None.
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            snr_range (List[float], optional): SNR range for the noising. Defaults to [-3.0, 5.0].
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
        self.snr_range = snr_range
        
    def setup(self, stage: str = None) -> None:
        """
        Sets up the datasets.
        This method loads and processes the datasets for training, validation, and testing.
        It renames, selects, casts, and formats the necessary columns as per the configuration.

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
            
        speech_noisy = speech_noisy.rename_column(f"audio.{self.sensor}", "audio_body_conducted")
        speech_noisy = speech_noisy.select_columns(["audio_body_conducted"])
        speech_noisy = speech_noisy.cast_column(
            "audio_body_conducted", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        speech_noisy = speech_noisy.with_format("torch")

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
            # speech_noisy_real = concatenate_datasets([speech_noisy["train"], speech_noisy["validation"], speech_noisy["test"]])
            
            # self.test_dataset = speech_noisy_real   
            
            # for this PR, speech_noisy_synthetic is used instead of speech_noisy_real
            #TODO: for next PR, use speech_noisy_real + speech_noisy_synthetic
            speech_test = speechclean["test"]
            noise_test = speechless_noisy["test"]
            
            self.test_dataset = SpeechNoiseDataset(speech_test, noise_test)

    def train_dataloader(self) -> DataLoader:
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

    def val_dataloader(self) -> DataLoader:
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

    def test_dataloader(self) -> DataLoader:
        """
        Test dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True, collate_strategy=self.collate_strategy
                ),
        )

    def data_collator(self, batch: List[Dict[str, Audio]], deterministic: bool, collate_strategy: str) -> Dict[str, torch.Tensor]:
        """
        Custom data collator function to mix speech and noise audios and dynamically pad the data.

        This function processes a batch of data by mixing clean speech with noise at specified SNR ranges.
        It then pads or trims the audio signals based on the collate strategy and applies data augmentation
        if specified.

        Args:
            - batch (List[Dict[str, Audio]]): Dict from the dataset with the keys :
                - 'audio_body_conducted': (torch.Tensor of dimension (sample_rate * duration)),
                - 'audio_airborne': (torch.Tensor of dimension (sample_rate * duration))
                - 'audio_body_conducted_speechless_noisy': (torch.Tensor of dimension (sample_rate * noise_duration))
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
        noise_batch = [item["audio_body_conducted_speechless_noisy"]["array"] for item in batch] # len(noise_batch) > len(body_conducted_batch)
        
        speech_noisy_synthetic, _ = mix_speech_and_noise(body_conducted_batch, noise_batch, self.snr_range)
        
        if collate_strategy == "pad":

            speech_noisy_synthetic_padded_batch = pad_sequence(
                speech_noisy_synthetic, batch_first=True, padding_value=0.0
            ).unsqueeze(1)
            air_conducted_padded_batch = pad_sequence(
                air_conducted_batch, batch_first=True, padding_value=0.0
            ).unsqueeze(1)

        else:
            ms_length = int(self.collate_strategy.split('-')[1])
            samples = int(self.sample_rate * ms_length / 1000)

            speech_noisy_synthetic_padded_batch = []
            air_conducted_padded_batch = []
            for body_conducted, air_conducted in zip(speech_noisy_synthetic, air_conducted_batch):
                body_conducted_padded, air_conducted_padded = set_audio_duration(
                    audio=body_conducted,
                    desired_samples=samples,
                    audio_bis=air_conducted,
                    deterministic=deterministic,
                )
                speech_noisy_synthetic_padded_batch.append(body_conducted_padded.unsqueeze(0))
                air_conducted_padded_batch.append(air_conducted_padded.unsqueeze(0))
            speech_noisy_synthetic_padded_batch = torch.stack(speech_noisy_synthetic_padded_batch, dim=0)
            air_conducted_padded_batch = torch.stack(air_conducted_padded_batch, dim=0)

        # Apply data augmentation
        if deterministic is False:
            with torch.no_grad():
                speech_noisy_synthetic_padded_batch, air_conducted_padded_batch = self.data_augmentation(speech_noisy_synthetic_padded_batch, air_conducted_padded_batch)        

        return {
            "audio_body_conducted": speech_noisy_synthetic_padded_batch,
            "audio_airborne":  air_conducted_padded_batch,
        }