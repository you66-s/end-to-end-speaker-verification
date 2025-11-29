import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SpeakerEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # --- Feature extractor (4 blocks CNN)
        self.features = nn.Sequential(
            ConvBlock(3, 32),   
            ConvBlock(32, 64),
            ConvBlock(64, 128), 
            ConvBlock(128, 256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output: 256 × 1 × 1
        self.embedding_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # Fully connected embedding
        x = self.embedding_fc(x)
        # 4) Normalize embeddings for triplet loss
        x = F.normalize(x, p=2, dim=1)
        return x
