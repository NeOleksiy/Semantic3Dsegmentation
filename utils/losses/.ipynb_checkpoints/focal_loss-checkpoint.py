import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, N) - логиты модели
        # targets: (B, N) - метки классов (0..C-1)
        
        # Преобразуем метки в one-hot формат
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()  # (B, N, C)
        targets_onehot = targets_onehot.permute(0, 2, 1)  # (B, C, N)
        
        # Вычисляем вероятности с помощью softmax
        probs = F.softmax(inputs, dim=1)  # (B, C, N)
        
        # Вычисляем кросс-энтропию
        ce_loss = -targets_onehot * torch.log(probs + 1e-6)  # (B, C, N)
        
        # Вычисляем focal term
        focal_term = (1 - probs) ** self.gamma
        
        # Применяем focal term
        focal_loss = focal_term * ce_loss  # (B, C, N)
        
        # Применяем веса классов (alpha)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor([self.alpha] * inputs.size(1)).to(inputs.device)
            else:
                alpha = self.alpha.to(inputs.device)
            focal_loss = alpha.view(1, -1, 1) * focal_loss
        
        # Игнорирование определенного класса
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)  # (B, 1, N)
            focal_loss = focal_loss * mask.float()
        
        # Агрегация потерь
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss