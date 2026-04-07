import torch
import torch.nn as nn
import torchaudio

class AudioEncoder(nn.Module):
    """
    Extracts MFCC features and applies a small CNN to get a 512-dim embedding.
    Input: (batch_size, 1, T_audio) raw waveform (resampled to 16kHz)
    Output: (batch_size, 512)
    """
    def __init__(self, sample_rate=16000, n_mfcc=40, output_dim=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        # MFCC extraction (fixed, non-trainable)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 64}
        )

        # CNN to process MFCC sequence (n_mfcc, time)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),  # (128, 128) → time dimension becomes 128
            nn.Flatten(),
            nn.Linear(128 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, waveform):
        """
        waveform: (batch, 1, T_audio)
        """
        # Compute MFCC (batch, 1, n_mfcc, time)
        mfcc = self.mfcc_transform(waveform)  # (batch, 1, n_mfcc, T)
        print(f"MFCC shape before squeeze: {mfcc.shape}")  # Debug
        # Remove the channel dimension (since it's always 1 for mono)
        mfcc = mfcc.squeeze(1)                 # (batch, n_mfcc, T)
        print(f"MFCC shape after squeeze: {mfcc.shape}")   # Debug
        # Pass through CNN
        emb = self.cnn(mfcc)                    # (batch, output_dim)
        return emb