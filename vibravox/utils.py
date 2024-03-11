"""Various utilities."""

import torch
from torchaudio.functional import lowpass_biquad


def set_audio_duration(
        audio: torch.Tensor, desired_time_len: int, deterministic: bool = False
):
    """
    Make the audio signal have the desired duration.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_len)
        desired_time_len (int): duration of selected signal in number of time steps
        deterministic (bool): if True, always select the same part of the signal

    """
    original_time_len = audio.shape[-1]

    # If the signal is longer than the desired duration, select a random part of the signal
    if original_time_len >= desired_time_len:
        if deterministic:
            offset_time_len = original_time_len - desired_time_len // 2
        else:
            offset_time_len = torch.randint(
                low=0, high=original_time_len - desired_time_len + 1, size=(1,)
            )
        audio = audio[..., offset_time_len: offset_time_len + desired_time_len]

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


def remove_hf(waveform: torch.Tensor, sample_rate: int, cutoff_freq: int, padding_length: int = 3000):
    """
    Low-pass filter of the fourth order with zero-phase shift.

    Args:
        waveform (torch.Tensor): input signal, shape: (..., time_len)
        sample_rate (int): sample rate of the input signal in Hz
        cutoff_freq (int): cut-off frequency in Hz
        padding_length (int): length of the padding for the IRR response stabilisation
    """
    # pad for IRR response stabilisation
    pad = torch.nn.ReflectionPad1d(padding_length)
    waveform = pad(waveform)

    # filt-filt trick for 0-phase shift

    def lowpass_filter(x):
        return lowpass_biquad(
            x, sample_rate=sample_rate, cutoff_freq=cutoff_freq
        )

    def reverse(x):
        return torch.flip(input=x, dims=[-1])

    waveform = reverse(lowpass_filter(reverse(lowpass_filter(waveform))))

    # un-pad
    waveform = waveform[..., padding_length: -padding_length]

    return waveform
