import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) for aligning two modalities.
    Used for video-text, video-audio, text-audio pairs.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (batch_size, emb_dim) normalized embeddings.
        Returns symmetric contrastive loss.
        """
        batch_size = z_i.size(0)
        # Compute cosine similarity matrix
        sim = torch.matmul(z_i, z_j.T) / self.temperature   # (batch, batch)
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=sim.device)
        loss_i = self.criterion(sim, labels)
        loss_j = self.criterion(sim.T, labels)
        return (loss_i + loss_j) / 2.0