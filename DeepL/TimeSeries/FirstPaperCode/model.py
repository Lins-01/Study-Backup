import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerPredictor(nn.Module):
    def __init__(self, num_features, num_timesteps, num_classes, num_layers=6):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(num_features, 8, dim_feedforward=512),
            num_layers=num_layers
        )
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)

        x = self.transformer_encoder(x)
        # x shape: (batch_size, seq_len, num_features)

        x = x[:, -1, :]
        # take the last time step

        x = self.linear(x)
        # shape: (batch_size, num_classes)

        return x


model = TransformerPredictor(num_features=32,
                             num_timesteps=64,
                             num_classes=10)

# Example usage:
batch_x = torch.randn(64, 64, 32)
batch_y = model(batch_x)
# print(batch_y)
print(model)