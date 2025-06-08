import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """3D Residual Block with two convolutional layers and a skip connection.
     
    Args:
        in_channels: Number of input/output channels
    """
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        features = self.conv_layers(x)
        return F.relu(residual + features, inplace=True)