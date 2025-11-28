"""

This demo script showcases real-time speech enhancement using the "Cnam-LMSSC/EBEN_throat_microphone" model:

--> https://huggingface.co/Cnam-LMSSC/EBEN_throat_microphone

This a non-causal version of EBEN that we make causal by waiting for enough future context
which corresponds to half of the receptive field of the model (207ms).

"""

import torch
import sounddevice as sd
import numpy as np
import threading
import sys
import termios
import tty
from vibravox.torch_modules.dnn.eben_generator import EBENGenerator

# ======================================================
# KEYBOARD LISTENER (non-blocking)
# ======================================================
eben_enabled = False  # default ON


def keyboard_listener():
    global eben_enabled
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            ch = sys.stdin.read(1)
            if ch == "e":
                eben_enabled = True
                print("🔊 EBEN ENABLED: 207ms latency")
            elif ch == "d":
                eben_enabled = False
                print("🔇 EBEN DISABLED (passthrough): 16ms latency")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
listener_thread.start()

# ======================================================
# EBEN + AUDIO SETUP
# ======================================================
sample_rate = 16_000
hop_size = 256  # (=16ms) minimal number of samples needed to produce a new chunk of output given enough input samples
window_size = 6624  # (=414ms) first valid length greater than 6340 (model receptive field) + 256 (hop_size)
# the model receptive field can be computed from the architecture or simply by feeding a very long signal
# of zeros except at one position and checking the length of the output that is non-zero

# Valid region within the output window
valid_start = (window_size - hop_size) // 2
valid_end = (window_size + hop_size) // 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = EBENGenerator.from_pretrained("Cnam-LMSSC/EBEN_throat_microphone")
model = model.eval().to(device)

in_stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=hop_size,
    dtype="float32",
)
out_stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=hop_size,
    dtype="float32",
)

in_stream.start()
out_stream.start()

print("🎤 Live enhancement running... CTRL+C to stop")
print("Press 'e' to ENABLE EBEN, 'd' to DISABLE EBEN")

# Rolling buffers
in_buffer = torch.zeros(1, 1, 0, device=device)
out_buffer = np.zeros(0, dtype=np.float32)

try:
    while True:
        # (1) Read throat microphone input (for mic ref: https://vibravox.cnam.fr/documentation/hardware/sensors/throat)
        in_block, _ = in_stream.read(hop_size)
        block_np = in_block.T
        block_t = torch.from_numpy(block_np).unsqueeze(0).to(device)

        if not eben_enabled:
            out_stream.write(in_block)  # direct pass-through
            continue

        # (2) Append to rolling buffer
        in_buffer = torch.cat([in_buffer, block_t], dim=-1)

        # (3) Process windows if enough samples
        while in_buffer.shape[-1] >= window_size:
            input_chunk = in_buffer[:, :, :window_size]

            with torch.no_grad():
                enhanced_chunk, _ = model(input_chunk)

            valid = enhanced_chunk[:, :, valid_start:valid_end]
            valid_np = valid.squeeze().detach().cpu().numpy().astype(np.float32)

            out_buffer = np.concatenate([out_buffer, valid_np])
            in_buffer = in_buffer[:, :, hop_size:]

        # (4) Play output
        if out_buffer.shape[0] >= hop_size:
            play_chunk = out_buffer[:hop_size]
            out_buffer = out_buffer[hop_size:]
        else:
            play_chunk = np.zeros(hop_size, dtype=np.float32)

        out_stream.write(play_chunk.reshape(-1, 1))

except KeyboardInterrupt:
    print("\n🛑 Exiting...")

finally:
    in_stream.stop()
    in_stream.close()
    out_stream.stop()
    out_stream.close()
