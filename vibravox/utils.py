"""Various utilities."""

import torch
from torchaudio.functional import lowpass_biquad


from typing import Optional

def pad_audio(audio: torch.Tensor, desired_time_len: int) -> torch.Tensor:
    """
    Pad the audio tensor to the desired length. The padding is done symmetrically.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_len)
        desired_time_len (int): desired duration of the signal in number of time steps

    Returns:
        torch.Tensor: Padded audio tensor.
    """

    assert audio.shape[-1] <= desired_time_len, "The audio signal is longer than the desired duration. Use set_audio_duration instead."

    original_time_len = audio.shape[-1]
    num_zeros_left = desired_time_len - original_time_len // 2
    return torch.nn.functional.pad(
        audio,
        pad=(
            num_zeros_left,
            desired_time_len - original_time_len - num_zeros_left,
        ),
        mode="constant",
        value=0,
    )

def slice_audio(audio: torch.Tensor, desired_time_len: int, offset_time_len: int) -> torch.Tensor:
    """
    Slice the audio tensor to the desired length.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_len)
        desired_time_len (int): desired duration of the signal in number of time steps
        offset_time_len (int): offset for slicing the audio tensor

    Returns:
        torch.Tensor: Sliced audio tensor.
    """

    assert audio.shape[-1] >= desired_time_len, "The audio signal is shorter than the desired duration. Use pad_audio instead."

    return audio[..., offset_time_len: offset_time_len + desired_time_len]

def set_audio_duration(
    audio: torch.Tensor, desired_time_len: int, audio_bis: Optional[torch.Tensor] = None, deterministic: bool = False
) -> torch.Tensor:
    """
    Make the audio signal have the desired duration.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_len)
        desired_time_len (int): duration of selected signal in number of time steps
        audio_bis (torch.Tensor, optional): second input signal with the same shape: (..., time_len)
        deterministic (bool): if True, always select the same part of the signal

    Returns:
        torch.Tensor: Audio tensor with the desired duration.
    """
    original_time_len = audio.shape[-1]

    assert audio_bis is None or audio.shape == audio_bis.shape, "The two audio signals must have the same shape."

    if original_time_len >= desired_time_len:
        offset_time_len = original_time_len - desired_time_len // 2 if deterministic else torch.randint(
            low=0, high=original_time_len - desired_time_len + 1, size=(1,)
        )
        audio = slice_audio(audio, desired_time_len, offset_time_len)
        if audio_bis is not None:
            audio_bis = slice_audio(audio_bis, desired_time_len, offset_time_len)
    else:
        audio = pad_audio(audio, desired_time_len)
        if audio_bis is not None:
            audio_bis = pad_audio(audio_bis, desired_time_len)

    return (audio, audio_bis) if audio_bis is not None else audio


def remove_hf(
    waveform: torch.Tensor,
    sample_rate: int,
    cutoff_freq: int,
    padding_length: int = 3000,
):
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
        return lowpass_biquad(x, sample_rate=sample_rate, cutoff_freq=cutoff_freq)

    def reverse(x):
        return torch.flip(input=x, dims=[-1])

    waveform = reverse(lowpass_filter(reverse(lowpass_filter(waveform))))

    # un-pad
    waveform = waveform[..., padding_length:-padding_length]

    return waveform
