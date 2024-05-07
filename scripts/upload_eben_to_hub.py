import torch
from vibravox.torch_modules.dnn.eben_generator import EBENGenerator

SENSOR = "body_conducted.in_ear.rigid_earpiece_microphone"
PATH = f"outputs/run/bwe/{SENSOR}/2024-05-07_14-58-32_p2_q3_chan24/checkpoints/epoch_001.ckpt"

checkpoint = torch.load(PATH)

# Clean state_dict
state_dict = {k.replace('generator.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('generator')}


# Infer model hparams from checkpoint
m = state_dict['pqmf.analysis_weights'].shape[0]
n = state_dict['pqmf.analysis_weights'].shape[2]
p = state_dict['first_conv.weight'].shape[1]

# Load model
model = EBENGenerator(m=m, n=n, p=p)
model.load_state_dict(state_dict)

model.push_to_hub(f"Cnam-LMSSC/EBEN_{SENSOR}", commit_message=f"Upload EBENGenerator")