from typing import List, Any, Dict, Union
from torch.utils.data import Dataset
from random import randint

class SpeechNoiseDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    
    def __init__(self, speech_dataset: Dataset, noise_dataset: Dataset):
        
        super().__init__()
        
        self.speech_dataset = speech_dataset
        self.noise_dataset = noise_dataset
        self.len_noise: int = len(noise_dataset)
        
    def __len__(self) -> int:
        return len(self.speech_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        speech_sample = self.speech_dataset[idx]
        # Randomize the noise sample
        idx = randint(0, self.len_noise-1)
        noise_sample = self.noise_dataset[idx]
        
        return {
            "audio_airborne": speech_sample["audio_airborne"],
            "audio_body_conducted": speech_sample["audio_body_conducted"],
            "audio_body_conducted_speechless_noisy": noise_sample["audio_body_conducted_speechless_noisy"],
        }