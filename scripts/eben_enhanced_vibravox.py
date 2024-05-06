# TODO: finish to implement the script

import torch
from datasets import load_dataset

from vibravox.torch_modules.dnn.eben_generator import EBENGenerator

ebens = {"forehead": EBENGenerator(4, 4, 4), #EBENGenerator.from_pretrained(f"Cnam-LMSSC/path_to_eben")
         "rigid_inear": EBENGenerator(4, 4, 4),
         "soft_inear": EBENGenerator(4, 4, 4),
         "throat": EBENGenerator(4, 4, 4),
         "temple": EBENGenerator(4, 4, 4)}

test_dataset = load_dataset("Cnam-LMSSC/vibravox", "speech_clean", split="test", streaming=False)


def enhance_audio(sample):

    for sensor, eben in ebens.items():
        audio = sample[f"audio.{sensor}"]
        cut_audio = eben.cut_to_valid_length(audio)
        enhanced_audio = eben(cut_audio)
        sample[f"audio.{sensor}"] = enhanced_audio

    return sample


enhanced_test_dataset = test_dataset.map(enhance_audio)


