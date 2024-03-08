"""Various utilities."""

import torch

def set_audio_duration(
        audio: torch.Tensor, desired_time_len: int, deterministic: bool = False
):
    """
    Make the audio signal have the desired duration.

    Args:
        audio (torch.Tensor): input signal. Shape: (time_len,)
        desired_time_len (int): duration of selected signal in number of time steps
        deterministic (bool): if True, always select the same part of the signal

    """
    original_time_len = audio.numel()

    # If the signal is longer than the desired duration, select a random part of the signal
    if original_time_len >= desired_time_len:
        if deterministic:
            offset_time_len = original_time_len - desired_time_len // 2
        else:
            offset_time_len = torch.randint(
                low=0, high=original_time_len - desired_time_len + 1, size=(1,)
            )
        audio = audio[offset_time_len: offset_time_len + desired_time_len]

    # If the signal is shorter than the desired duration, pad the signal with zeros
    else:
        num_zeros_left = desired_time_len - original_time_len // 2
        audio = torch.nn.functional.pad(
            audio,
            pad=(
                num_zeros_left,
                desired_time_len - original_time_len - num_zeros_left,
            ),
            mode="constant",
            value=0,
        )

    return audio
