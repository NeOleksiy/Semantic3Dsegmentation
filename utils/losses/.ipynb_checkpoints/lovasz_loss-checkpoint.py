import torch
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1e-7)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax(preds, labels, ignore_index=-1):

    batch_size, num_classes, n_points = preds.shape
    
    # Применяем softmax и преобразуем в вероятности
    probas = F.softmax(preds, dim=1)
    
    # Преобразуем в плоский формат
    probas_flat = probas.permute(1, 0, 2).reshape(num_classes, -1)  # (C, B*N)
    labels_flat = labels.reshape(-1)  # (B*N,)
    
    # Фильтрация игнорируемых точек
    mask = labels_flat != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
        
    probas_flat = probas_flat[:, mask]  # (C, M)
    labels_flat = labels_flat[mask]     # (M,)
    
    # Находим присутствующие классы в батче
    present_classes = torch.unique(labels_flat)
    losses = []
    
    for c in present_classes:
        fg = (labels_flat == c).float()  # Фронтграунд для класса c
        class_probas = probas_flat[c]    # Вероятности класса c
        
        # Вычисляем ошибки и сортируем
        errors = (fg - class_probas).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        
        # Вычисляем градиент Lovasz
        grad = lovasz_grad(fg_sorted)
        
        # Скалярное произведение ошибок и градиента
        class_loss = torch.dot(errors_sorted, grad)
        losses.append(class_loss)
    
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=preds.device)