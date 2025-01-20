from typing import Dict
import torch
from torch.utils.data import Dataset
from datasets import Audio

class SpeechNoiseDataset(Dataset):
    """
    A dataset that pairs speech samples with randomized noise samples.

    This dataset takes a speech dataset and a noise dataset, and for each speech sample,
    it randomly selects a noise sample to mix with, providing a combination of clean speech 
    and noisy speech samples.

    Args:
        speech_dataset (Dataset): The dataset containing clean speech samples.
        noise_dataset (Dataset): The dataset containing noise samples.
    """
    
    def __init__(self, speech_dataset: Dataset, noise_dataset: Dataset):
        super().__init__()
        
        self.speech_dataset: Dataset = speech_dataset
        self.noise_dataset: Dataset = noise_dataset
        self.len_noise: int = len(noise_dataset)
        
    def __len__(self) -> int:
        """
        Return the total number of speech samples in the dataset.

        Returns:
            int: The number of speech samples.
        """
        return len(self.speech_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Audio]]:
        """
        Retrieve the paired speech and noise samples at the specified index.

        This method fetches a speech sample from the speech dataset using the provided index.
        It then randomly selects a noise sample from the noise dataset to pair with the speech sample.

        Args:
            idx (int): The index of the speech sample to retrieve.

        Returns:
            Dict[str, Dict[str, Audio]]: Dict from the dataset with the keys :
                - 'audio_airborne': The airborne audio speech sample of dimension (sample_rate * duration).
                - 'audio_body_conducted': The body-conducted audio speech sample of dimension (sample_rate * duration).
                - 'audio_body_conducted_speechless_noisy': The body-conducted noisy speech sample of dimension (sample_rate * noise_duration).
        """
        speech_sample = self.speech_dataset[idx]
        # Randomize the noise sample
        idx = torch.randint(0, self.len_noise, (1,)).item()
        noise_sample = self.noise_dataset[idx]
        
        return {
            "audio_airborne": speech_sample["audio_airborne"],
            "audio_body_conducted": speech_sample["audio_body_conducted"],
            "audio_body_conducted_speechless_noisy": noise_sample["audio_body_conducted_speechless_noisy"],
        }