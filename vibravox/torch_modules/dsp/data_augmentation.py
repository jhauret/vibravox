from typing import Tuple

import torch
import torchaudio.transforms as T
from vibravox.torch_modules.dsp.time_masking_waveform import TimeMaskingBlockWaveform


class WaveformDataAugmentation(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        p_data_augmentation=0,
        p_speed_perturbation=0.3,
        p_pitch_shift=0.3,
        p_time_masking=0.3,
        speed_perturbation_factors=(0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3),
        pitch_shift_steps=(-4, -3, -2, -1, 1, 2, 3, 4, 5, 6),
        time_masking_percentage=(1, 2, 3, 4, 5, 6, 7, 8),
    ):
        super().__init__()

        self.sample_rate = sample_rate

        assert 0 <= p_data_augmentation <= 1, "p_data_augmentation must be in [0, 1]"
        assert 0 <= p_speed_perturbation <= 1, "p_speed_perturbation must be in [0, 1]"
        assert 0 <= p_pitch_shift <= 1, "p_pitch_shift must be in [0, 1]"
        assert 0 <= p_time_masking <= 1, "p_time_masking must be in [0, 1]"

        self.apply_data_augmentation = p_data_augmentation
        self.p_speed_perturbation = p_speed_perturbation
        self.p_pitch_shift = p_pitch_shift
        self.p_time_masking = p_time_masking

        self.speed_perturbation_factors = speed_perturbation_factors
        self.pitch_shift_steps = pitch_shift_steps
        self.time_masking_percentage = time_masking_percentage

    def forward(self, waveform_1: torch.Tensor, waveform_2: torch.Tensor = None) -> Tuple[torch.Tensor]:

        """
        Args:
            waveform_1:  first input waveform tensor of shape (..., time)
            waveform_2:  second input waveform tensor of shape (..., time), can be ignored

        Returns:
            Tuple[torch.Tensor]: first and second augmented waveform tensor of shape (..., time)
        """

        if torch.rand(1) < self.apply_data_augmentation:
            # Apply data augmentation
            if torch.rand(1) < self.p_speed_perturbation:
                # Apply speed perturbation
                speed_perturbation_factor = self.speed_perturbation_factors[torch.randint(len(self.speed_perturbation_factors), size=(1,)).item()]
                speed_perturbation = T.SpeedPerturbation(orig_freq=self.sample_rate, factors=[speed_perturbation_factor])
                # This is tricky but SpeedPerturbation returns two values even when lengths is None
                waveform_1, _ = speed_perturbation(waveform_1)
                if waveform_2 is not None: waveform_2, _ = speed_perturbation(waveform_2)
            if torch.rand(1) < self.p_pitch_shift:
                # Apply pitch shift
                pitch_shift_step = self.pitch_shift_steps[torch.randint(len(self.pitch_shift_steps), size=(1,)).item()]
                pitch_shift = T.PitchShift(self.sample_rate, n_steps=pitch_shift_step)
                waveform_1 = pitch_shift(waveform_1)
                if waveform_2 is not None: waveform_2 = pitch_shift(waveform_2)
            if torch.rand(1) < self.p_time_masking:
                # Apply time masking
                time_masking_percentage = self.time_masking_percentage[torch.randint(len(self.time_masking_percentage), size=(1,)).item()]
                time_masking = TimeMaskingBlockWaveform(masking_percentage=time_masking_percentage)
                waveform_1 = time_masking(waveform_1)
                if waveform_2 is not None: waveform_2 = time_masking(waveform_2)

        return waveform_1, waveform_2
