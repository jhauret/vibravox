"""Various utilities."""

import torch
from torchaudio.functional import lowpass_biquad


from typing import Optional, Tuple

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
    speech_batch: torch.Tensor, noise_batch: torch.Tensor, snr: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mix speech and noise batches at a given Signal-to-Noise Ratio (SNR).

    Args:
        speech_batch (torch.Tensor): A batch of clean speech samples with shape (batch_size, 1, time).
        noise_batch (torch.Tensor): A batch of noise samples with shape (noise_batch_size, 1, time_noise).
        snr (float, optional): Desired signal-to-noise ratio in decibels. Defaults to 5.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - corrupted_speech_batch: Mixed speech and noise with shape (batch_size, 1, time).
            - noise_batch_scaled: Scaled noise used for mixing with shape (batch_size, 1, time).

    Raises:
        TypeError: If either speech_batch or noise_batch is not a torch.Tensor.
        ValueError: If tensors do not have the expected number of dimensions or incompatible shapes.
    """
    # -----------------------------
    # Input Validation with Assertions
    # -----------------------------
    # Check if inputs are torch.Tensor
    if not isinstance(speech_batch, torch.Tensor):
        raise TypeError(f"speech_batch must be a torch.Tensor, but got {type(speech_batch)}")
    if not isinstance(noise_batch, torch.Tensor):
        raise TypeError(f"noise_batch must be a torch.Tensor, but got {type(noise_batch)}")

    # Check tensor dimensions
    if speech_batch.dim() != 3:
        raise ValueError(f"speech_batch must be a 3D tensor of shape (batch_size, 1, time), but got shape {speech_batch.shape}")
    if noise_batch.dim() != 3:
        raise ValueError(f"noise_batch must be a 3D tensor of shape (noise_batch_size, 1, time_noise), but got shape {noise_batch.shape}")

    batch_size, _, time_speech = speech_batch.size()
    noise_batch_size, _, time_noise = noise_batch.size()

    # -----------------------------
    # Align Time Sizes by Shuffling Chunks
    # -----------------------------
    # For each speech sample, randomly select a noise sample and a random offset
    device = speech_batch.device  # Ensure all tensors are on the same device

    # Randomly select noise sample indices for each speech sample
    noise_indices = torch.randint(0, noise_batch_size, (batch_size,), device=device)

    # Calculate maximum possible offset
    max_offset = time_noise - time_speech

    # Randomly select offsets for each speech sample
    offsets = torch.randint(0, max_offset + 1, (batch_size,), device=device)

    # Extract the corresponding noise slices
    noise_slices = []
    for i in range(batch_size):
        noise_sample = noise_batch[noise_indices[i], :, offsets[i]:offsets[i] + time_speech]
        noise_slices.append(noise_sample)
    noise_slices = torch.stack(noise_slices, dim=0)  # Shape: (batch_size, 1, time_speech)

    # -----------------------------
    # Compute Power
    # -----------------------------
    # Compute power (mean squared amplitude) for each sample in the batch
    speech_power = torch.mean(speech_batch ** 2, dim=-1, keepdim=True)  # Shape: (batch_size, 1, 1)
    noise_power = torch.mean(noise_slices ** 2, dim=-1, keepdim=True)    # Shape: (batch_size, 1, 1)

    # -----------------------------
    # Compute Scaling Factor
    # -----------------------------
    snr_linear = 10 ** (snr / 10.0)  # Convert SNR from dB to linear scale
    scale_factor = torch.sqrt(speech_power / (noise_power * snr_linear))  # Shape: (batch_size, 1, 1)

    # -----------------------------
    # Scale Noise and Mix
    # -----------------------------
    noise_batch_scaled = noise_slices * scale_factor  # Scale noise to achieve desired SNR
    corrupted_speech_batch = speech_batch + noise_batch_scaled  # Mix scaled noise with speech

    return corrupted_speech_batch, noise_batch_scaled