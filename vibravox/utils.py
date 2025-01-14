"""Various utilities."""

import torch
from torchaudio.functional import lowpass_biquad
from typing import List, Tuple, Optional

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

def decode_operations(predicted_chr: str,
                      label_chr: str,
                      editops: List[Tuple[str, int, int]]) -> List[Tuple[str, str, str]]:
    """
    Decode the operations based on the edit operations.

    Args:
        predicted_chr (str): The predicted character.
        label_chr (str): The label character.
        editops (List[Tuple[str, int, int]]): The list of edit operations.

    Returns:
        List[Tuple[str, str, str]]: The list of decoded operations.
    """
    ops = []
    for editop in editops:
        op, pred_idx, label_idx = editop

        if op == "insert":
            label_token = label_chr[label_idx]
            ops.append((op, label_token, label_token))
        elif op == "delete":
            pred_token = predicted_chr[pred_idx]
            ops.append((op, pred_token, pred_token))
        else:
            label_token = label_chr[label_idx]
            pred_token = predicted_chr[pred_idx]
            ops.append((op, pred_token, label_token))

    return ops


def get_space_indices(string: str) -> List[int]:
    """
    Get the positions of spaces in a string.

    Args:
        string (str): The input string.

    Returns:
        List[int]: The list of space indices.
    """
    return [i for i, x in enumerate(string) if x == ' ']


def split_editops(pred: str,
                  target: str,
                  editops: List[Tuple[str, int, int]])\
        -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    """
    Split the edit operations into three categories: before space, in word, and all.

    Args:
        pred (str): The predicted string.
        target (str): The target string.
        editops (List[Tuple[str, int, int]]): The list of edit operations.

    Returns:
        Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], List[Tuple[str, int, int]]]: The split edit operations.
    """
    pred_space_idx = get_space_indices(pred)
    target_space_idx = get_space_indices(target)

    raw_editops_before_space = []
    raw_editops_in_word = []
    for editop in editops:
        op, pred_idx, label_idx = editop

        if ((op == 'replace' and ((pred_idx+1) in pred_space_idx or (label_idx + 1) in target_space_idx)) or
            (op == 'delete' and (pred_idx+1) in pred_space_idx) or
            (op == 'insert' and (label_idx + 1) in target_space_idx)):
            raw_editops_before_space.append(editop)
        else:
            raw_editops_in_word.append(editop)

    return raw_editops_before_space, raw_editops_in_word, editops