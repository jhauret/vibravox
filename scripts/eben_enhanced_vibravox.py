import torch
import torchaudio
from datasets import load_dataset

from vibravox.torch_modules.dnn.eben_generator import EBENGenerator

EBENs = {"forehead_accelerometer": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_forehead_accelerometer"),
         "rigid_in_ear_microphone": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_rigid_in_ear_microphone"),
         "soft_in_ear_microphone": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_soft_in_ear_microphone"),
         "throat_microphone": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_throat_microphone"),
         "temple_vibration_pickup": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_temple_vibration_pickup")}

resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

test_dataset = load_dataset("Cnam-LMSSC/vibravox", "speech_clean", split="test", streaming=False)


def enhance_audio(sample):
    for sensor, eben in EBENs.items():
        audio = torch.Tensor(sample[f"audio.{sensor}"]["array"])[None, None, ...]
        resampled_audio = resampler(audio)
        cut_audio = eben.cut_to_valid_length(resampled_audio)
        enhanced_audio, _ = eben(cut_audio)
        sample[f"audio.{sensor}"]["array"] = enhanced_audio.detach().numpy().squeeze()
        sample[f"audio.{sensor}"]["sampling_rate"] = 16_000

    return sample


enhanced_test_dataset = test_dataset.map(enhance_audio)
enhanced_test_dataset.push_to_hub(repo_id="Cnam-LMSSC/vibravox_enhanced_by_EBEN",
                                  config_name="speech_clean",
                                  commit_message="Enhance audio with EBEN trained on Vibravox")

