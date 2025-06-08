import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Вычисляет k ближайших соседей для каждого вектора в наборе
    
    Args:
        x: Входной тензор формы (batch_size, num_features, num_points)
        k: Количество соседей
        
    Returns:
        Индексы k ближайших соседей формы (batch_size, num_points, k)
    """
    inner_product = torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
    square_norm = torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
    
    # Формула: ||a - b||^2 = ||a||^2 - 2<a,b> + ||b||^2
    pairwise_distance = square_norm - 2 * inner_product + square_norm.transpose(2, 1)
    pairwise_distance = -pairwise_distance 
    
    neighbors_idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return neighbors_idx

def get_graph_feature(
    x: torch.Tensor, 
    k: int = 20, 
    idx: torch.Tensor = None,
    include_coordinates: bool = True
) -> torch.Tensor:
    """
    Формирует признаки ребер графа на основе kNN
    
    Args:
        x: Исходные точки [B, C, N]
        k: Количество соседей
        idx: Предвычисленные индексы соседей (опционально)
        include_coordinates: Флаг использования координат как базовых признаков
        
    Returns:
        Тензор признаков ребер [B, 2*C, N, k]
    """
    batch_size, num_dims, num_points = x.size()
    
    if include_coordinates and num_dims >= 3:
        base_features = x 
    else:
        base_features = x[:, :3, :]
    
    if idx is None:
        idx = knn(base_features, k=k)
    
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)  # [B*N*k]
    
    x_permuted = x.transpose(2, 1).contiguous()  # [B, N, C]
    neighbor_features = x_permuted.view(batch_size * num_points, -1)[idx, :]
    neighbor_features = neighbor_features.view(batch_size, num_points, k, num_dims)  # [B, N, k, C]
    
    central_points = x_permuted.view(batch_size, num_points, 1, num_dims)  # [B, N, 1, C]
    central_points = central_points.repeat(1, 1, k, 1)  # [B, N, k, C]
    
    # Формируем признаки ребер: [разность, центральная точка]
    edge_features = torch.cat([
        neighbor_features - central_points,  # Локальные признаки
        central_points                      # Глобальный контекст
    ], dim=-1)  # [B, N, k, 2*C]
    
    # [B, 2*C, N, k]
    return edge_features.permute(0, 3, 1, 2).contiguous()

class DGCNN(nn.Module):
    """Динамический графовый CNN для классификации точечных облаков
    
    Архитектура:
        - 5 динамических графовых слоев
        - Многоуровневое объединение признаков
        - Глобальный пулинг + skip-connections
        - 3 полносвязных слоя для классификации
    
    Args:
        num_classes: Количество выходных классов
        input_channels: Размерность входных признаков (по умолчанию 11)
        k: Количество соседей в графе (по умолчанию 20)
        emb_dims: Размерность глобального дескриптора (по умолчанию 1024)
        dropout: Вероятность dropout (по умолчанию 0.5)
    """
    def __init__(
        self,
        num_classes: int,
        extra_feature_channels: int = 3,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5
    ):
        super().__init__()
        self.k = k
        self.input_channels = 3 + extra_feature_channels

        bn_init = lambda channels: nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2 * input_channels, 64, kernel_size=1, bias=False),
            bn_init(64)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            bn_init(64)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            bn_init(64)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            bn_init(64)
        )
        
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            bn_init(64)
        )
        
        self.global_feature_extractor = nn.Sequential(
            nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Dropout(p=dropout),
            nn.Conv1d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass DGCNN
        
        Args:
            x: Входной тензор [B, C, N]
            
        Returns:
            Логиты классов [B, num_classes, N]
        """
        batch_size, _, num_points = x.size()
        
        edge_features = get_graph_feature(x, k=self.k, include_coordinates=True)
        x = self.conv_block1(edge_features)
        x = self.conv_block2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]  # [B, 64, N]
        
        edge_features = get_graph_feature(x1, k=self.k)
        x = self.conv_block3(edge_features)
        x = self.conv_block4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # [B, 64, N]
        
        edge_features = get_graph_feature(x2, k=self.k)
        x3 = self.conv_block5(edge_features).max(dim=-1, keepdim=False)[0]  # [B, 64, N]
        
        mid_level_features = torch.cat([x1, x2, x3], dim=1)  # [B, 192, N]
        
        global_descriptor = self.global_feature_extractor(mid_level_features)  # [B, emb_dims, N]
        global_descriptor = global_descriptor.max(dim=-1, keepdim=True)[0]  # [B, emb_dims, 1]
        
        global_context = global_descriptor.repeat(1, 1, num_points)  # [B, emb_dims, N]
        
        all_features = torch.cat([
            global_context,     # global
            x1, x2, x3,        # local
            mid_level_features  # mid
        ], dim=1)  # [B, 1216, N]
        
        return self.classifier(all_features)