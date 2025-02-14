import torch
from datasets import load_dataset

speech_clean = load_dataset("Cnam-LMSSC/vibravox", "speech_clean")
speechless_noisy = load_dataset("Cnam-LMSSC/vibravox", "speechless_noisy")

speech_clean = speech_clean.with_format("torch")
speechless_noisy = speechless_noisy.with_format("torch")

def mix_speech(example):
    rand_idx = torch.randint(0, len(speechless_noisy["test"]), (1,)).item()
    noise_tt = speechless_noisy["test"][rand_idx]

    speech_samples = example["audio.headset_microphone"]["array"].size(0)
    noise_samples = noise_tt["audio.headset_microphone"]["array"].size(0)

    start_time = torch.randint(0, noise_samples - speech_samples, (1,)).item()

    for mic in ["audio.headset_microphone",
                "audio.throat_microphone",
                "audio.soft_in_ear_microphone",
                "audio.rigid_in_ear_microphone",
                "audio.forehead_accelerometer",
                "audio.temple_vibration_pickup"]:
        noise_sliced = noise_tt[mic]["array"][start_time: start_time + speech_samples]
        example[mic]["array"] = example[mic]["array"] + noise_sliced

    return example

# Apply the transformation
speech_mixed = speech_clean["test"].map(mix_speech)

# Push to hub
speech_mixed.push_to_hub("Cnam-LMSSC/vibravox_mixed_for_spkv", "speech_noisy_mixed")

