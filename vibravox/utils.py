"""Various utilities."""

import torch

def set_audio_duration(
        sample_rate:int, audio: torch.Tensor, desired_duration: float, deterministic: bool = False
):
    """
    Make the audio signal have the desired duration.

    Args:
        sample_rate (int): sample rate of the audio signal
        audio (torch.Tensor): input signal. Shape: (time_len,)
        desired_duration (float): duration of selected signal in seconds
        deterministic (bool): if True, always select the same part of the signal

    """
    original_time_len = audio.numel()
    desired_time_len = int(desired_duration * sample_rate)

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
