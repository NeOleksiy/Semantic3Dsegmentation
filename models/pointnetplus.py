import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks.utils import sample_and_group_all, sample_and_group, square_distance, index_points


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SA) Module - downsampling layer
    Performs point sampling, local feature grouping and feature extraction
    
    Architecture:
    1. Samples points using Farthest Point Sampling (FPS)
    2. Groups points into local regions using ball query
    3. Applies shared MLP to extract local features
    4. Performs max pooling within each local region
    
    Args:
        npoint (int): Number of points to sample
        radius (float): Search radius for ball query grouping
        nsample (int): Max number of neighbors in each local region
        in_channel (int): Input feature dimension
        mlp (list): List of output dimensions for each MLP layer
        group_all (bool): Whether to group all points (global feature)
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint          # Number of output points
        self.radius = radius          # Ball query radius
        self.nsample = nsample        # Max neighbors per group
        self.group_all = group_all    # Global grouping flag
        
        # Build shared MLP layers
        self.mlp_convs = nn.ModuleList()  # 1x1 Conv layers
        self.mlp_bns = nn.ModuleList()    # BatchNorm layers
        last_channel = in_channel
        
        # Dynamically create MLP layers
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):

        # Rearrange dimensions for processing
        xyz = xyz.permute(0, 2, 1)  # (B, N, 3)
        if points is not None:
            points = points.permute(0, 2, 1)  # (B, N, D)

        # Sample and group points
        if self.group_all:
            # Global feature extraction
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # Local region feature extraction
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        
        # Feature extraction - shared MLP
        # new_points: (B, npoint, nsample, 3+D) -> (B, 3+D, nsample, npoint)
        new_points = new_points.permute(0, 3, 2, 1)  
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling over neighbors
        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], npoint)
        new_xyz = new_xyz.permute(0, 2, 1)  # (B, 3, npoint)
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation (FP) Module - upsampling layer
    Propagates features from sparser to denser point sets
    
    Key operations:
    1. Inverse distance weighted interpolation
    2. Skip connection with features from same level
    3. Feature refinement with shared MLP
    
    Args:
        in_channel (int): Input feature dimension
        mlp (list): List of output dimensions for each MLP layer
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        
        # Build shared MLP layers
        self.mlp_convs = nn.ModuleList()  # 1x1 Conv layers 
        self.mlp_bns = nn.ModuleList()    # BatchNorm layers
        last_channel = in_channel
        
        # Dynamically create MLP layers
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):

        # Rearrange dimensions for processing
        xyz1 = xyz1.permute(0, 2, 1)  # (B, N, 3)
        xyz2 = xyz2.permute(0, 2, 1)  # (B, S, 3)
        points2 = points2.permute(0, 2, 1)  # (B, S, D)
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # Feature interpolation
        if S == 1:
            # Single point case - replicate features
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Inverse distance weighted interpolation
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest neighbors
            
            # Compute interpolation weights
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            # Weighted sum of features
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        # Skip connection (if available)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # (B, N, D)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Feature refinement - shared MLP
        new_points = new_points.permute(0, 2, 1)  # (B, D', N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ Segmentation Network - hierarchical point feature learning
    
    Architecture:
    - Encoder: 4 Set Abstraction (SA) layers (downsampling)
    - Decoder: 4 Feature Propagation (FP) layers (upsampling)
    - Final per-point classification
    
    Args:
        num_classes (int): Number of segmentation classes
    """
    def __init__(self, num_classes, extra_feature_channels=3, dropout=0.5):
        super().__init__()
        
        # Encoder - Downsampling path
        # SA1: 1024 points, radius 0.1, 32 neighbors
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6+extra_feature_channels, [32, 32, 64], False)
        # SA2: 256 points, radius 0.2
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        # SA3: 64 points, radius 0.4 
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        # SA4: 16 points, radius 0.8 (global features)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        # Decoder - Upsampling path
        # FP4: 512 + 256 -> [256, 256]
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        # FP3: 256 + 128 -> [256, 256]
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        # FP2: 256 + 64 -> [256, 128]
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        # FP1: 128 + 0 -> [128, 128, 128]
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)  # Feature refinement
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)         # Regularization
        self.conv2 = nn.Conv1d(128, num_classes, 1)  # Class prediction

    def forward(self, xyz):

        l0_points = xyz          # (B, 6, N)
        l0_xyz = xyz[:, :3, :]  # (B, 3, N) - just coordinates
        
        # Encoder - hierarchical downsampling
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # (B, 3, 1024), (B, 64, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 256), (B, 128, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 64), (B, 256, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 16), (B, 512, 16)
        
        # Decoder - hierarchical upsampling
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       # (B, 128, N)
        
        # Segmentation prediction
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)  # (B, num_classes, N)
        x = F.log_softmax(x, dim=1)  # Log probabilities
        
        return x