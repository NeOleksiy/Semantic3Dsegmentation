import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock3D

class Voxelization(nn.Module):
    """A module for voxelization of point clouds with feature extraction.
        Features:
         - Normalization of coordinates to the range [0, 1]
         - Aggregation of points into voxels (average coordinates + density)
         - Voxel Feature Encoding (VFE) for extracting local features
     
        Args:
         grid_size: Voxel grid size (default 64)
    """
    def __init__(self, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        
        # Voxel Feature Encoder (VFE)
        self.vfe = nn.Sequential(
            nn.Linear(4, 64),          # XYZ + density -> 64x vector
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),  
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Converts a point cloud into a voxel representation
         
        Args:
            point_cloud: Tensor of shape (B, 3, N)
        
        Returns:
            Voxel features of the form (B, 128, grid_size, grid_size, grid_size)
        """
        batch_size, _, num_points = point_cloud.shape
        device = point_cloud.device

        # normalize
        coords = point_cloud.permute(0, 2, 1)  # -> (B, N, 3)
        coord_min, _ = torch.min(coords, dim=1, keepdim=True)
        coord_max, _ = torch.max(coords, dim=1, keepdim=True)
        normalized_coords = (coords - coord_min) / (coord_max - coord_min + 1e-6)
        
        voxel_indices = (normalized_coords * (self.grid_size - 1)).long()
        
        # Voxel grid
        voxel_grid = torch.zeros(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.grid_size,
            4,  #  [mean_x, mean_y, mean_z, density]
            device=device
        )
        
        for batch_idx in range(batch_size):
            unique_voxels, inverse_indices, point_counts = torch.unique(
                voxel_indices[batch_idx],
                return_inverse=True,
                return_counts=True,
                dim=0
            )
            
            # Calculating the average coordinates for each voxel
            coord_sums = torch.zeros((len(unique_voxels), 3), device=device)
            coord_sums.scatter_add_(
                dim=0,
                index=inverse_indices.view(-1, 1).expand(-1, 3),
                src=coords[batch_idx]
            )
            mean_coords = coord_sums / point_counts.view(-1, 1).float()
            
            # Calculating density of coordinates
            density = point_counts.float() / num_points
            
            voxel_grid[batch_idx, 
                      unique_voxels[:, 0], 
                      unique_voxels[:, 1], 
                      unique_voxels[:, 2], 
                      :3] = mean_coords
            voxel_grid[batch_idx, 
                      unique_voxels[:, 0], 
                      unique_voxels[:, 1], 
                      unique_voxels[:, 2], 
                      3] = density
        
        voxel_features = self.vfe(voxel_grid.view(-1, 4))
        voxel_features = voxel_features.view(
            batch_size, 
            self.grid_size, 
            self.grid_size, 
            self.grid_size, 
            -1
        ).permute(0, 4, 1, 2, 3)  # -> (B, C, D, H, W)
        
        return voxel_features


class VoxelNet(nn.Module):
    """Network for processing voxelized point clouds.
    
    Architecture:
     - Voxelizer
     - Encoder with residual blocks and pools
     - Decoder with transoposed convolutions
     - Trilinear feature interpolation
     - Point classifier
     
    Args:
        num_classes: Number of classes to classify
        grid_size: Voxel grid size (default 64)
    """
    def __init__(self, num_classes: int = 20, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        
        self.voxelizer = Voxelization(grid_size)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            ResidualBlock3D(128),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            ResidualBlock3D(256),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(256, 512, kernel_size=3, padding=1, bias=False),
            ResidualBlock3D(512)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
            ResidualBlock3D(256),
            
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            ResidualBlock3D(128)
        )
        
        # Classificator
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )
        
        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:

        voxel_features = self.voxelizer(point_cloud)
        
        encoded = self.encoder(voxel_features)
        
        decoded = self.decoder(encoded)
        
        point_features = self._interpolate_features(decoded, point_cloud)
        
        return self.classifier(point_features)
    
    def _interpolate_features(
        self, 
        voxel_features: torch.Tensor, 
        point_cloud: torch.Tensor
    ) -> torch.Tensor:
        # Trilinear interpolation of voxel features at point positions
        batch_size, _, num_points = point_cloud.shape
        
        normalized_points = point_cloud.permute(0, 2, 1)  # (B, N, 3)
        coord_min, _ = torch.min(normalized_points, dim=1, keepdim=True)
        coord_max, _ = torch.max(normalized_points, dim=1, keepdim=True)
        normalized_points = 2 * (normalized_points - coord_min) / (coord_max - coord_min + 1e-6) - 1
        
        sample_points = normalized_points.view(batch_size, 1, 1, num_points, 3)
        
        point_features = F.grid_sample(
            voxel_features,
            sample_points,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )
        
        return point_features.squeeze(2).squeeze(2)  # (B, C, N)