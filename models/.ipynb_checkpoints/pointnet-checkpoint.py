import torch
import torch.nn as nn


class TNet(nn.Module):
    def __init__(self, dim, num_points=10000):
        super().__init__()

        self.dim = dim
        self.layer1 = self._make_block_mlp(dim, 64)
        self.layer2 = self._make_block_mlp(64, 128)
        self.layer3 = self._make_block_mlp(128, 1024)

        self.max_pool = nn.MaxPool1d(num_points)
        
        self.layer4 = self._make_block_fc(1024, 512)
        self.layer5 = self._make_block_fc(512, 256)
        self.fc1 = nn.Linear(256, dim**2)
    
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
        bs = x.shape[0]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.max_pool(x).view(bs, -1)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)

        identity = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)

        if x.is_cuda:
            identity = identity.cuda()

        x = x.view(-1, self.dim, self.dim) + identity

        return x

class PointNetBackbone(nn.Module):
    def __init__(self, num_points=10000, num_global_features=1024):
        super().__init__()
        self.num_points = num_points

        self.TNet_1 = TNet(dim=3, num_points=num_points)
        self.TNet_2 = TNet(dim=64, num_points=num_points)

        self.layer1 = self._make_block(3, 64)
        self.layer2 = self._make_block(64, 64)

        self.layer3 = self._make_block(64, 64)
        self.layer4 = self._make_block(64, 128)
        self.layer5 = self._make_block(128, num_global_features)

        self.max_pool =nn.MaxPool1d(num_points, return_indices=True)

    def _make_block(self, in_dim, out_dim, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        bs = x.shape[0]

        A_input = self.TNet_1(x)

        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        x = self.layer1(x)
        x = self.layer2(x)

        A_feat = self.TNet_2(x)

        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1) 

        local_features = x.clone()

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        featrues = torch.cat((local_features, 
                              global_features.unsqueeze(-1).repeat(1, 1, self.num_points)),
                              dim=1)
        return featrues, critical_indexes, A_feat

class PointNet(nn.Module):
    def __init__(self, num_points=10000, num_global_features=1024, num_classes=20):
        super().__init__()

        self.backbone = PointNetBackbone(num_points, num_global_features)

        num_features = num_global_features + 64

        self.layer1 = self._make_block(num_features, 512)
        self.layer2 = self._make_block(512, 256)
        self.layer3 = self._make_block(256, 128)
        self.fc1 = nn.Conv1d(128, num_classes, 1)
    
    def _make_block(self, in_dim, out_dim, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x, critical_indexes, A_feat = self.backbone(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc1(x)

        # x = x.transpose(2, 1)

        return x