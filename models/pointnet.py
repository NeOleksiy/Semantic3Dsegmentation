import torch
import torch.nn as nn


import torch
import torch.nn as nn

class TNet(nn.Module):
    """
    Transformation Network (T-Net) learns a transformation matrix for input features.
    Used to make PointNet invariant to geometric transformations.
    
    Args:
        dim (int): Dimension of input features (3 for spatial, 64 for feature space)
        num_points (int): Number of points in point cloud (default: 10000)
    """
    def __init__(self, dim, num_points=10000):
        super().__init__()
        self.dim = dim
        
        # Feature extraction layers (MLP)
        self.layer1 = self._make_block_mlp(dim, 64)
        self.layer2 = self._make_block_mlp(64, 128)
        self.layer3 = self._make_block_mlp(128, 1024)
        
        # Global max pooling
        self.max_pool = nn.MaxPool1d(num_points)
        
        # Fully connected layers to predict transformation matrix
        self.layer4 = self._make_block_fc(1024, 512)
        self.layer5 = self._make_block_fc(512, 256)
        self.fc1 = nn.Linear(256, dim**2)  # Output dim x dim transformation matrix
    
    def _make_block_mlp(self, in_dim, out_dim, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def _make_block_fc(self, in_dim, out_dim, kernel_size=1):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global feature vector
        x = self.max_pool(x).view(batch_size, -1)
        
        # Predict transformation matrix
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)
        
        # Initialize identity matrix and add to predicted transformation
        identity = torch.eye(self.dim, requires_grad=True).repeat(batch_size, 1, 1)
        
        if x.is_cuda:
            identity = identity.cuda()
            
        x = x.view(-1, self.dim, self.dim) + identity
        
        return x


class PointNetBackbone(nn.Module):
    """
    PointNet backbone network for feature extraction.
    
    Args:
        num_points (int): Number of points in point cloud (default: 10000)
        num_global_features (int): Dimension of global features (default: 1024)
    """
    def __init__(self, num_points=10000, num_global_features=1024):
        super().__init__()
        self.num_points = num_points
        
        # Transformation networks
        self.TNet_1 = TNet(dim=3, num_points=num_points)  # For input transform
        self.TNet_2 = TNet(dim=64, num_points=num_points)  # For feature transform
        
        # Feature extraction layers
        self.layer1 = self._make_block(3, 64)
        self.layer2 = self._make_block(64, 64)
        
        self.layer3 = self._make_block(64, 64)
        self.layer4 = self._make_block(64, 128)
        self.layer5 = self._make_block(128, num_global_features)
        
        # Global max pooling (returns both values and indices)
        self.max_pool = nn.MaxPool1d(num_points, return_indices=True)
    
    def _make_block(self, in_dim, out_dim, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Input transformation
        A_input = self.TNet_1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        
        # First feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Feature transformation
        A_feat = self.TNet_2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        
        # Save local features before further processing
        local_features = x.clone()
        
        # Further feature extraction
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        # Global max pooling
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(batch_size, -1)
        critical_indexes = critical_indexes.view(batch_size, -1)
        
        # Combine local and global features
        features = torch.cat((
            local_features,
            global_features.unsqueeze(-1).repeat(1, 1, self.num_points)
        ), dim=1)
        
        return features, critical_indexes, A_feat


class PointNet(nn.Module):
    """
    Complete PointNet architecture for point cloud classification.
    
    Args:
        num_points (int): Number of points in point cloud (default: 10000)
        num_global_features (int): Dimension of global features (default: 1024)
        num_classes (int): Number of output classes (default: 20)
    """
    def __init__(self, num_points=10000, num_global_features=1024, num_classes=20):
        super().__init__()
        
        # Backbone network for feature extraction
        self.backbone = PointNetBackbone(num_points, num_global_features)
        
        # Classification head
        num_features = num_global_features + 64  # Global + local features
        
        self.layer1 = self._make_block(num_features, 512)
        self.layer2 = self._make_block(512, 256)
        self.layer3 = self._make_block(256, 128)
        self.fc1 = nn.Conv1d(128, num_classes, 1)  # Final classification layer
    
    def _make_block(self, in_dim, out_dim, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Extract features from backbone
        x, critical_indexes, A_feat = self.backbone(x)
        
        # Classification head
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc1(x)
        
        return x