import torch
import re
from typing import Dict, List, Tuple
from lightning import LightningDataModule
from datasets import Audio, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from vibravox.utils import mix_speech_and_noise_without_rescaling
from vibravox.utils import set_audio_duration
from vibravox.torch_modules.dsp.data_augmentation import WaveformDataAugmentation
from vibravox.datasets.speech_noise import SpeechNoiseDataset

class NoisyBWELightningDataModule(LightningDataModule):
    
    def __init__(
        self,
        sample_rate: int = 16000,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        sensor: str = "headset_microphone",
        collate_strategy: str = "constant_length-2500-ms",
        data_augmentation: torch.nn.Module = None,
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        """
        LightningDataModule for Noisy Bandwidth Extension (BWE)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name (str, optional): Dataset name.
                Must be one of "Cnam-LMSSC/vibravox" or "Cnam-LMSSC/vibravox_enhanced_by_EBEN".
                Defaults to "Cnam-LMSSC/vibravox".
            sensor (str, optional): Sensor. Defaults to "headset_microphone"
            collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - "pad": Pad the audio signals to the length of the longest signal in the batch.
                - "constant_length-XXX-ms": Cut or pad the audio signals to XXXms.
            Defaults to "constant_length-2500-ms".
            data_augmentation (nn.Module, optional): Data augmentation module. Defaults to None.
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        assert dataset_name in ["Cnam-LMSSC/vibravox",
                                "Cnam-LMSSC/vibravox2",
                                "Cnam-LMSSC/vibravox-test",
                                "Cnam-LMSSC/vibravox_mixed_for_spkv",
                                "Cnam-LMSSC/vibravox_enhanced_by_EBEN"], \
            f"dataset_name {dataset_name} not supported."
        self.dataset_name = dataset_name
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
        self.pin_memory = pin_memory
        
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
            self.dataset_name, "speech_clean", streaming=self.streaming
        )
        speechless_noisy = load_dataset(
            self.dataset_name, "speechless_noisy", streaming=self.streaming
        )
        speech_noisy = load_dataset(
            self.dataset_name, "speech_noisy", streaming=self.streaming
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
            
            self.train_dataset_synthetic = SpeechNoiseDataset(speech_train, noise_train)
            self.val_dataset_synthetic = SpeechNoiseDataset(speech_validation, noise_validation)
            
            self.val_dataset_real = speech_noisy["validation"]
            
        if stage in ["test", None]:            
            speech_test = speechclean["test"]
            noise_test = speechless_noisy["test"]
            
            self.test_dataset_synthetic = SpeechNoiseDataset(speech_test, noise_test)
            
            self.test_dataset_real = speech_noisy["test"]

    def train_dataloader(self) -> DataLoader:
        """
        Train dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.train_dataset_synthetic,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(batch, deterministic=False, collate_strategy=self.collate_strategy),
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Dict[str, DataLoader]:
        """
        Validation dataloaders.

        Returns:
            Dict[str, DataLoader]
        """        
        dataloader_synthetic = DataLoader(
            self.val_dataset_synthetic,
            batch_size=min(1, self.batch_size // 4),
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                batch, deterministic=True, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )
        dataloader_real = DataLoader(
            self.val_dataset_real,
            batch_size=min(1, self.batch_size // 4),
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                batch, deterministic=True, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )

        return {"synthetic": dataloader_synthetic, "real": dataloader_real}

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """
        Test dataloaders.

        Returns:
            Dict[str, DataLoader]
        """    
        dataloader_synthetic = DataLoader(
            self.test_dataset_synthetic,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                batch, deterministic=True, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )
        dataloader_real = DataLoader(
            self.test_dataset_real,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                batch, deterministic=True, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )

        return {"synthetic": dataloader_synthetic, "real": dataloader_real}

    def data_collator(self, batch: List[Dict[str, Audio]], deterministic: bool, collate_strategy: str) -> Dict[str, torch.Tensor]:
        """
        Custom data collator function to mix speech and noise audios and dynamically pad the data.

        This function processes a batch of data by mixing clean speech with noise.
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
        
        if not "audio_airborne" in batch[0]:
            # collate strategy pad for noisy real data
            speech_noisy_real_padded_batch = pad_sequence(
                body_conducted_batch, batch_first=True, padding_value=0.0
            ).unsqueeze(1)
            return {"audio_body_conducted": speech_noisy_real_padded_batch}
        
        air_conducted_batch = [item["audio_airborne"]["array"] for item in batch]
        noise_batch = [item["audio_body_conducted_speechless_noisy"]["array"] for item in batch] # len(noise_batch) > len(body_conducted_batch)
        
        speech_noisy_synthetic, _ = mix_speech_and_noise_without_rescaling(body_conducted_batch, noise_batch)
        
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