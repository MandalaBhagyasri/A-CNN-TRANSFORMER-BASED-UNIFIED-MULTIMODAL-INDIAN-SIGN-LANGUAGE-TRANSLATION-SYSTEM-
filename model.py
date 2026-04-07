import torch
import torch.nn as nn
from torchvision.models import resnet18

class VideoTransformerModel(nn.Module):
    """
    CNN (ResNet18) + Transformer Encoder + Classification head.
    Input:  (B, T, C, H, W) with T=16, C=3, H=112, W=112
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=3):
        super().__init__()
        # CNN encoder per frame
        resnet = resnet18(pretrained=True)
        # Remove avgpool and fc, keep all conv layers, then add adaptive avgpool
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-2],           # up to last conv layer
            nn.AdaptiveAvgPool2d((1, 1))             # output (B, 512, 1, 1)
        )
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Combine batch and time for CNN processing
        cnn_in = x.view(B * T, C, H, W)               # (B*T, C, H, W)
        cnn_out = self.cnn(cnn_in)                    # (B*T, 512, 1, 1)
        cnn_out = cnn_out.squeeze(-1).squeeze(-1)     # (B*T, 512)

        # Reshape back to (B, T, 512)
        frame_features = cnn_out.view(B, T, -1)       # (B, T, 512)

        # Transformer expects (seq_len, batch, features)
        frame_features = frame_features.permute(1, 0, 2)  # (T, B, 512)

        encoded = self.transformer_encoder(frame_features)  # (T, B, 512)

        # Mean pooling over time
        pooled = encoded.mean(dim=0)                  # (B, 512)

        # Classification
        logits = self.fc(pooled)                      # (B, num_classes)
        return logits

# Quick test (uncomment to run)
if __name__ == '__main__':
    model = VideoTransformerModel(num_classes=10)
    dummy_input = torch.randn(1, 16, 3, 112, 112)     # batch size 1
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")