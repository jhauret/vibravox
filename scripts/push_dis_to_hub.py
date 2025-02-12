import torch
from vibravox.torch_modules.dnn.eben_discriminator import DiscriminatorEBENMultiScales

name = "Cnam-LMSSC/DiscriminatorEBENMultiScales_throat_microphone"

dis = DiscriminatorEBENMultiScales(q=4, min_channels=24)

state_dict = torch.load("/home/jhauret/Downloads/bwe/throat_microphone/2024-06-15_14-26-02/checkpoints/last.ckpt")["state_dict"]

for key in list(state_dict.keys()):
    if key.startswith("discriminator."):
        state_dict[key[14:]] = state_dict.pop(key)
    else:
        state_dict.pop(key)

dis.load_state_dict(state_dict, strict=True)

dis.push_to_hub(name, commit_message=f"Upload")

dis = DiscriminatorEBENMultiScales.from_pretrained(name)