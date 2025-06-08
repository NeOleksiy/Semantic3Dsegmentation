import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class OptimizedVoxelization(nn.Module):
    """Optimized voxelization module with dynamic dot count support.
     
    Features:
        - Automatic processing of any number of points at the entrance
        - Limitation of the maximum number of points in a voxel
        - Efficient calculation of average features through scatter operations
        - RGB feature support (7 input channels)
    
    Args:
        grid_size (int): Voxel grid size (default 64)
        max_points_per_voxel (int): Maximum number of points in a voxel (35 by default)
    """
    def __init__(self, grid_size: int = 64, max_points_per_voxel: int = 35, extra_feature_channels: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.max_points = max_points_per_voxel
        
        # Voxel Feature Encoder
        self.vfe = nn.Sequential(
            nn.Linear(4+extra_feature_channels, 64),          # XYZ (3) + density (1) + RGB (3)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:

        batch_size, _, num_points = point_cloud.shape
        device = point_cloud.device
        
        # normalize
        coords = point_cloud[:, :3].permute(0, 2, 1)  # (B, N, 3)
        coords_min = coords.amin(dim=1, keepdim=True)
        coords_max = coords.amax(dim=1, keepdim=True)
        norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-7)
        
        voxel_idx = (norm_coords * (self.grid_size - 1)).clamp(0, self.grid_size - 1 - 1e-4).long()
        
        batch_voxels = torch.zeros(
            batch_size, 128, self.grid_size, self.grid_size, self.grid_size,
            device=device
        )
        
        # Calculate voxel grid by batch
        for batch_idx in range(batch_size):
            unique_voxels, inverse_indices = torch.unique(
                voxel_idx[batch_idx], 
                dim=0,
                return_inverse=True,
                return_counts=False
            )
            
            valid_mask = (unique_voxels >= 0).all(dim=1) & (unique_voxels < self.grid_size).all(dim=1)
            valid_voxels = unique_voxels[valid_mask]
            
            if len(valid_voxels) == 0:
                continue

            remap = torch.full((len(unique_voxels),), -1, device=device)
            remap[valid_mask] = torch.arange(len(valid_voxels), device=device)
            valid_inverse = remap[inverse_indices]
            valid_points_mask = valid_inverse != -1
            
            mean_coords = scatter_mean(
                coords[batch_idx][valid_points_mask],
                valid_inverse[valid_points_mask],
                dim=0
            )
            
            density = torch.bincount(
                valid_inverse[valid_points_mask],
                minlength=len(valid_voxels)
            ).float().clamp(max=self.max_points) / self.max_points
            
            # mean RGB
            if point_cloud.size(1) >= 6:
                mean_rgb = scatter_mean(
                    point_cloud[batch_idx, 3:6].permute(1, 0)[valid_points_mask],
                    valid_inverse[valid_points_mask],
                    dim=0
                )
            else:
                mean_rgb = torch.zeros(len(valid_voxels), 3, device=device)
            
            # coord + density + rgb
            voxel_feat = torch.cat([
                mean_coords, 
                density.unsqueeze(-1), 
                mean_rgb
            ], dim=-1)
            
            voxel_features = self.vfe(voxel_feat).T  # (128, num_voxels)
            
            valid_voxels = valid_voxels.clamp(0, self.grid_size - 1)
            batch_voxels[batch_idx, :, valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]] = voxel_features
        
        return batch_voxels


class EnhancedVoxelNetPlus(nn.Module):
    """Improved 3D segmentation network with full U-Net architecture.
    
    Features:
        - Full encoder-decoder architecture with skip connections
        - Support for a dynamic number of points at the input
        - Optimized voxelization
        - Trilinear feature interpolation
    
    Args:
        num_classes (int): Number of classes for segmentation
        grid_size (int): Voxel grid size (default 64)
    """
    def __init__(self, num_classes: int = 20,
                 grid_size: int = 64,
                 max_points_per_voxel: int = 35,
                 extra_feature_channels: int = 3,
                 dropout: int = 0.5
                ):
        super().__init__()
        self.grid_size = grid_size
        
        self.voxelizer = OptimizedVoxelization(grid_size, max_points_per_voxel, extra_feature_channels)
        
        # Encoder
        self.enc1 = self._make_conv_block(128, 64)  # После вокселизации 128 каналов
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._make_conv_block(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._make_conv_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip-connections
        self.up1 = self._make_deconv_block(512, 256)
        self.dec1 = self._make_conv_block(512, 256)  # enc3 (256) + up1(256)
        
        self.up2 = self._make_deconv_block(256, 128)
        self.dec2 = self._make_conv_block(256, 128)  # enc2 (128) + up2(128)
        
        self.up3 = self._make_deconv_block(128, 64)
        self.dec3 = self._make_conv_block(128, 64)   # enc1 (64) + up3(64)
        
        self.final_conv = nn.Conv3d(64, 32, 1)
        
        self.classifier = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, 1)
        )

        self._init_weights()

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_deconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        # Initialization of Kaiming weights for convolutions and constant for BatchNorm.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:

        voxel_features = self.voxelizer(point_cloud)
        
        enc1 = self.enc1(voxel_features)       # [B,64,S,S,S]
        enc2 = self.enc2(self.pool1(enc1))     # [B,128,S/2,S/2,S/2]
        enc3 = self.enc3(self.pool2(enc2))     # [B,256,S/4,S/4,S/4]
        
        bottleneck = self.bottleneck(self.pool2(enc3))  # [B,512,S/8,S/8,S/8]
        
        up1 = self.up1(bottleneck)
        dec1 = self.dec1(torch.cat([up1, enc3], dim=1))
        
        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, enc1], dim=1))
        
        final_features = self.final_conv(dec3)  # [B,32,S,S,S]
        
        batch_size, _, num_points = point_cloud.size()
        points = point_cloud[:, :3].permute(0, 2, 1)  # [B,N,3]

        coords_min = points.amin(dim=1, keepdim=True)
        coords_max = points.amax(dim=1, keepdim=True)
        normalized_points = (points - coords_min) / (coords_max - coords_min + 1e-7)
        normalized_points = 2 * normalized_points - 1  # [-1, 1]
        
        point_features = F.grid_sample(
            final_features,
            normalized_points.view(batch_size, 1, 1, num_points, 3),
            align_corners=True,
            mode='bilinear'
        ).squeeze(2).squeeze(2)  # [B,32,N]
        
        return self.classifier(point_features)  # [B,num_classes,N]