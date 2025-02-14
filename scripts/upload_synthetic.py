import torch
from copy import deepcopy
from datasets import load_dataset
from tqdm import tqdm

speech_clean = load_dataset("Cnam-LMSSC/vibravox", "speech_clean")
speechless_noisy = load_dataset("Cnam-LMSSC/vibravox", "speechless_noisy")
speech_mixed = deepcopy(speech_clean["test"])
speech_clean = speech_clean.with_format("torch")
speechless_noisy = speechless_noisy.with_format("torch")
speech_mixed = speech_mixed.with_format("torch")

for idx, speech_utt in tqdm(enumerate(speech_clean["test"])):
    rand_idx = torch.randint(0, len(speechless_noisy["test"]), (1,)).item()
    noise_tt = speechless_noisy["test"][rand_idx]

    speech_samples = speech_utt["audio.headset_microphone"]["array"].size(0)
    noise_samples = noise_tt["audio.headset_microphone"]["array"].size(0)

    # Randomize noise segment
    start_time = torch.randint(0, noise_samples - speech_samples, (1,)).item()

    for mic in ["audio.headset_microphone",
                "audio.throat_microphone",
                "audio.soft_in_ear_microphone",
                "audio.rigid_in_ear_microphone",
                "audio.forehead_accelerometer",
                "audio.temple_vibration_pickup"]:

        noise_sliced = noise_tt[mic]["array"][start_time: start_time + speech_samples]
        speech_mixed[idx][mic]["array"]  = speech_utt[mic]["array"] + noise_sliced

speech_mixed.push_to_hub("Cnam-LMSSC/vibravox_mixed_for_spkv", "speech_noisy_mixed")
