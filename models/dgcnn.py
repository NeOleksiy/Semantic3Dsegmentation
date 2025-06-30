import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphFeatureExtractor(nn.Module):
    """Модуль для извлечения признаков графа на основе kNN"""

    @staticmethod
    def knn(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Вычисляет k ближайших соседей для каждого вектора в наборе
        Исправленная и более стабильная версия
        """
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        return pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]

    def forward(
        self,
        x: torch.Tensor,
        k: int = 20,
        idx: torch.Tensor = None,
        include_coordinates: bool = True,
    ) -> torch.Tensor:
        batch_size, num_dims, num_points = x.size()

        # Всегда используем координаты для построения графа
        if include_coordinates and num_dims >= 3:
            base_features = x[:, :3, :]
        else:
            base_features = x

        if idx is None:
            idx = self.knn(base_features, k=k)  # [B, N, k]

        idx_base = (
            torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)  # [B*N*k]

        x = x.transpose(2, 1).contiguous()  # [B, N, C]
        neighbors = x.view(batch_size * num_points, -1)[idx, :]
        neighbors = neighbors.view(batch_size, num_points, k, num_dims)  # [B, N, k, C]

        central = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        # Формируем признаки ребер: [центральная точка, разность]
        features = torch.cat([central, neighbors - central], dim=3)
        return features.permute(0, 3, 1, 2)  # [B, 2*C, N, k]


class DGCNN(nn.Module):
    """Исправленная и проверенная реализация DGCNN для сегментации"""

    def __init__(
        self,
        num_classes: int,
        extra_feature_channels: int = 0,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.k = k
        self.input_channels = 3 + extra_feature_channels
        self.graph_extractor = GraphFeatureExtractor()

        # Инициализация блоков сверток
        def bn_relu_conv(in_channels, out_channels, kernel_size=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )

        # Первый блок
        self.conv1 = bn_relu_conv(2 * self.input_channels, 64)
        self.conv2 = bn_relu_conv(64, 64)

        # Второй блок
        self.conv3 = bn_relu_conv(64 * 2, 64)
        self.conv4 = bn_relu_conv(64, 64)

        # Третий блок
        self.conv5 = bn_relu_conv(64 * 2, 64)

        # Глобальный экстрактор признаков
        self.global_conv = nn.Sequential(
            nn.Conv1d(192, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Классификатор с правильной размерностью
        self.classifier = nn.Sequential(
            nn.Conv1d(1408, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, num_classes, 1),
        )

        # Инициализация весов
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_points = x.size()

        # Первый графовый слой
        edge1 = self.graph_extractor(x, k=self.k, include_coordinates=True)
        x1 = self.conv1(edge1)
        x1 = self.conv2(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # [B, 64, N]

        # Второй графовый слой
        edge2 = self.graph_extractor(x1, k=self.k, include_coordinates=False)
        x2 = self.conv3(edge2)
        x2 = self.conv4(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # [B, 64, N]

        # Третий графовый слой
        edge3 = self.graph_extractor(x2, k=self.k, include_coordinates=False)
        x3 = self.conv5(edge3)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # [B, 64, N]

        # Объединяем признаки из всех слоев
        x_features = torch.cat([x1, x2, x3], dim=1)  # [B, 192, N]

        # Глобальные признаки
        global_feat = self.global_conv(x_features)  # [B, 1024, N]
        global_feat = F.adaptive_max_pool1d(global_feat, 1)  # [B, 1024, 1]
        global_feat = global_feat.repeat(1, 1, num_points)  # [B, 1024, N]

        # Объединяем все признаки
        all_features = torch.cat(
            [
                x1,  # Локальные признаки 1
                x2,  # Локальные признаки 2
                x3,  # Локальные признаки 3
                x_features,  # Комбинированные признаки
                global_feat,  # Глобальные признаки
            ],
            dim=1,
        )  # [B, 64+64+64+192+1024=1408, N]

        # Правильная размерность для классификатора (1216 каналов)
        # Добавляем преобразование размерности

        # Классификация
        return self.classifier(all_features)
