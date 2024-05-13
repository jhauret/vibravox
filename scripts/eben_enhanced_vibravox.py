import torch
import torchaudio
from datasets import load_dataset

from vibravox.torch_modules.dnn.eben_generator import EBENGenerator

EBENs = {"body_conducted.forehead.miniature_accelerometer": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_body_conducted.forehead.miniature_accelerometer"),
         "body_conducted.in_ear.rigid_earpiece_microphone": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_body_conducted.in_ear.rigid_earpiece_microphone"),
         "body_conducted.in_ear.comply_foam_microphone": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_body_conducted.in_ear.comply_foam_microphone"),
         "body_conducted.throat.piezoelectric_sensor": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_body_conducted.throat.piezoelectric_sensor"),
         "body_conducted.temple.contact_microphone": EBENGenerator.from_pretrained(f"Cnam-LMSSC/EBEN_body_conducted.temple.contact_microphone")}

resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)


test_dataset = load_dataset("Cnam-LMSSC/vibravox", "speech_clean", split="test", streaming=False)


def enhance_audio(sample):
    for sensor, eben in EBENs.items():
        audio = torch.Tensor(sample[f"audio.{sensor}"]["array"])[None, None, ...]
        resampled_audio = resampler(audio)
        cut_audio = eben.cut_to_valid_length(resampled_audio)
        enhanced_audio, _ = eben(cut_audio)
        sample[f"audio.{sensor}"]["array"] = enhanced_audio.detach().numpy().squeeze()
        sample[f"audio.{sensor}"]["sampling_rate"] = 16_000  # todo: this don't work, update sampling rate in dataset features

    return sample


enhanced_test_dataset = test_dataset.map(enhance_audio)
enhanced_test_dataset.push_to_hub(repo_id="Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp",
                                  config_name="speech_clean",
                                  commit_message="Enhance audio with EBEN trained on vibravox 140/200 participants")

