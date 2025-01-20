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
    speech_batch: List[torch.Tensor], noise_batch: List[torch.Tensor], snr_range: List[float] = [-3.0, 5.0]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Mixes clean speech samples with noise samples at randomized Signal-to-Noise Ratios (SNRs).

    This function takes batches of clean speech and noise tensors, applies random SNRs within a specified range
    to the noise, scales the noise accordingly, and mixes it with the speech to produce corrupted speech samples.
    The SNR for each noise segment is sampled uniformly from the provided `snr_range`.

    Args:
        speech_batch (List[torch.Tensor]): 
            A list of clean speech samples. Each tensor should be 1-dimensional with shape `(time,)`.
        noise_batch (List[torch.Tensor]): 
            A list of noise samples. Each tensor should be 1-dimensional with shape `(time_noise,)`.
            The length of each noise sample must be greater than or equal to the corresponding speech sample.
        snr_range (List[float], optional): 
            A list containing two floats representing the minimum and maximum SNR values in decibels (dB).
            Defaults to `[-3.0, 5.0]`.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]:
            - `corrupted_speech_batch`: 
                A list of corrupted speech samples obtained by adding scaled noise to the clean speech.
                Each tensor has the same shape as the corresponding input speech tensor `(time,)`.
            - `noise_batch_scaled`: 
                A list of scaled noise tensors used for mixing. Each tensor has the same shape as the 
                corresponding input speech tensor `(time,)`.
    """
    # Input validation
    if not isinstance(speech_batch, list) or not all(isinstance(t, torch.Tensor) for t in speech_batch):
        raise TypeError("speech_batch must be a list of torch.Tensor")
    if not isinstance(noise_batch, list) or not all(isinstance(t, torch.Tensor) for t in noise_batch):
        raise TypeError("noise_batch must be a list of torch.Tensor")
    if len(speech_batch) != len(noise_batch):
        raise ValueError("speech_batch and noise_batch must have the same length")

    corrupted_speech_batch: List[torch.Tensor] = []
    noise_batch_scaled: List[torch.Tensor] = []

    number_segments: int = 10

    for speech, noise in zip(speech_batch, noise_batch):
        
        # Compute power
        speech_power = torch.mean(speech ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if speech.dim() != 1:
            raise ValueError(f"Each speech sample must be a 1D tensor, but got shape {speech.shape}")
        if noise.dim() != 1:
            raise ValueError(f"Each noise sample must be a 1D tensor, but got shape {noise.shape}")

        time_speech = speech.size(0)
        time_noise = noise.size(0)

        if time_noise < time_speech:
            raise ValueError(f"noise_sample length ({time_noise}) must be >= speech_sample length ({time_speech})")
               
        # Trim noise to fit into equal segments
        total_length = (len(noise) // number_segments) * number_segments
        noise = noise[:total_length].view(number_segments, -1)
        
        # Randomize segment order and SNRs
        permutation = torch.randperm(number_segments)
        snrs = torch.empty(number_segments).uniform_(snr_range[0], snr_range[1])
        
        # Compute scaling factor
        snrs_linear = 10 ** (snrs / 10.0)
        scale_factor = torch.sqrt(speech_power / (noise_power * snrs_linear)).unsqueeze(1)
        
        noise_scaled = noise[permutation] * scale_factor
        mixed_noise = noise_scaled.flatten()[:time_speech]

        # Scale noise and mix
        corrupted_speech = speech + mixed_noise

        corrupted_speech_batch.append(corrupted_speech)
        noise_batch_scaled.append(noise_scaled)

    return corrupted_speech_batch, noise_batch_scaled