import torch
from vibravox.torch_modules.dnn.eben_generator import EBENGenerator

SENSOR = "body_conducted.in_ear.comply_foam_microphone"
RUN = "2024-05-11_05-29-31_p2_q4_chan24"
CHECKPOINT = "last.ckpt"

PATH = f"outputs/run/bwe/{SENSOR}/{RUN}/checkpoints/{CHECKPOINT}"

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

model.push_to_hub(f"Cnam-LMSSC/EBEN_{SENSOR}", commit_message=f"Upload EBENGenerator trained on vibravox_140/200_participants")