"""Various utilities."""

import torch
from torchaudio.functional import lowpass_biquad


from typing import List, Optional, Tuple

def pad_audio(audio: torch.Tensor, desired_samples: int) -> torch.Tensor:
    """
    Pad the audio tensor to the desired length. The padding is done symmetrically.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_samples)
        desired_samples (int): desired duration of the signal in number of time samples

    Returns:
        torch.Tensor: Padded audio tensor.
    """

    assert audio.shape[-1] <= desired_samples, "The audio signal is longer than the desired duration. Use set_audio_duration instead."

    initial_samples = audio.shape[-1]
    num_zeros_left = desired_samples - initial_samples // 2
    return torch.nn.functional.pad(
        audio,
        pad=(
            num_zeros_left,
            desired_samples - initial_samples - num_zeros_left,
        ),
        mode="constant",
        value=0,
    )

def slice_audio(audio: torch.Tensor, desired_samples: int, offset_samples: int) -> torch.Tensor:
    """
    Slice the audio tensor to the desired length.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_samples)
        desired_samples (int): desired duration of the signal in number of time steps
        offset_samples (int): offset for slicing the audio tensor

    Returns:
        torch.Tensor: Sliced audio tensor.
    """

    assert audio.shape[-1] >= desired_samples, "The audio signal is shorter than the desired duration. Use pad_audio instead."

    return audio[..., offset_samples: offset_samples + desired_samples]

def set_audio_duration(
    audio: torch.Tensor, desired_samples: int, audio_bis: Optional[torch.Tensor] = None, deterministic: bool = False
) -> torch.Tensor:
    """
    Make the audio signal have the desired duration.

    Args:
        audio (torch.Tensor): input signal. Shape: (..., time_samples)
        desired_samples (int): duration of selected signal in number of time steps
        audio_bis (torch.Tensor, optional): second input signal with the same shape: (..., time_samples)
        deterministic (bool): if True, always select the same part of the signal

    Returns:
        torch.Tensor: Audio tensor with the desired duration.
    """
    initial_samples = audio.shape[-1]

    assert audio_bis is None or audio.shape == audio_bis.shape, "The two audio signals must have the same shape."

    if initial_samples >= desired_samples:
        offset_samples = (initial_samples - desired_samples) // 2 if deterministic else torch.randint(
            low=0, high=initial_samples - desired_samples + 1, size=(1,)
        )
        audio = slice_audio(audio, desired_samples, offset_samples)
        if audio_bis is not None:
            audio_bis = slice_audio(audio_bis, desired_samples, offset_samples)
    else:
        audio = pad_audio(audio, desired_samples)
        if audio_bis is not None:
            audio_bis = pad_audio(audio_bis, desired_samples)

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
        waveform (torch.Tensor): input signal, shape: (..., time_samples)
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

def mix_speech_and_noise(
    speech_batch: List[torch.Tensor], noise_batch: List[torch.Tensor], snr: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mix speech and noise at a given Signal-to-Noise Ratio (SNR).

    Args:
        speech_batch (torch.Tensor): A clean speech sample with shape (time).
        noise_batch (torch.Tensor): A noise sample with shape (time_noise).
        snr (float, optional): Desired signal-to-noise ratio in decibels. Defaults to 5.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - corrupted_speech_batch: Mixed speech and noise with shape (time).
            - noise_batch_scaled: Scaled noise used for mixing with shape (time).

    Raises:
        TypeError: If either speech_batch or noise_batch is not a torch.Tensor.
        ValueError: If tensors do not have the expected number of dimensions or incompatible shapes.
    """
    # -----------------------------
    # Input Validation with Assertions
    # -----------------------------

    # Check tensor dimensions
    if speech_batch[-1].dim() != 1:
        raise ValueError(f"speech_batch must be a 1D tensor of shape (time), but got shape {speech_batch.shape}")
    if noise_batch[-1].dim() != 1:
        raise ValueError(f"noise_batch must be a 1D tensor of shape (time_noise), but got shape {noise_batch.shape}")
    
    speech_batch = torch.cat(speech_batch, dim=0)
    noise_batch = torch.cat(noise_batch, dim=0)

    time_speech = speech_batch.size(0)
    time_noise = noise_batch.size(0)

    if time_noise < time_speech:
        raise ValueError(f"noise_batch length ({time_noise}) must be >= speech_batch length ({time_speech}).")

    # -----------------------------
    # Align Time Sizes by Shuffling Chunks
    # -----------------------------
    # Randomly select an offset
    device = speech_batch.device  # Ensure tensors are on the same device

    max_offset = time_noise - time_speech

    offset = torch.randint(0, max_offset + 1, (1,), device=device).item()

    # Extract the noise slice
    noise_slice = noise_batch[offset: offset + time_speech]

    # -----------------------------
    # Compute Power
    # -----------------------------
    speech_power = torch.mean(speech_batch ** 2)
    noise_power = torch.mean(noise_slice ** 2)

    # -----------------------------
    # Compute Scaling Factor
    # -----------------------------
    snr_linear = 10 ** (snr / 10.0)  # Convert SNR from dB to linear scale
    scale_factor = torch.sqrt(speech_power / (noise_power * snr_linear))

    # -----------------------------
    # Scale Noise and Mix
    # -----------------------------
    noise_batch_scaled = noise_slice * scale_factor  # Scale noise to achieve desired SNR
    corrupted_speech_batch = speech_batch + noise_batch_scaled  # Mix scaled noise with speech

    return corrupted_speech_batch, noise_batch_scaled