import torch
import torchaudio.transforms as T
from vibravox.torch_modules.dsp.time_masking_waveform import TimeMaskingBlockWaveform


class WaveformDataAugmentation(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.pitch_shift = T.PitchShift(sample_rate, n_steps=4)
        self.time_masking = TimeMaskingBlockWaveform(masking_percentage=5)
        self.speed_perturbation = T.SpeedPerturbation(orig_freq=16000, factors=[0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2])
        # waveform speed will be adjusted by factor sampled uniformly from factors

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:

        """
        Args:
            waveform:  input waveform tensor of shape (batch_size, num_channels, num_samples)

        Returns:
            torch.Tensor: augmented waveform tensor of shape (batch_size, num_channels, num_samples)
        """

        if torch.rand(1) < 0.3:
            waveform = self.pitch_shift(waveform)
        if torch.rand(1) < 0.3:
            waveform = self.time_masking(waveform)
        if torch.rand(1) < 0.3:
            waveform = self.speed_perturbation(waveform)

        return waveform
