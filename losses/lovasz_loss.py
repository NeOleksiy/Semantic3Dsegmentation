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

    probas = F.softmax(preds, dim=1)

    probas_flat = probas.permute(1, 0, 2).reshape(num_classes, -1)  # (C, B*N)
    labels_flat = labels.reshape(-1)  # (B*N,)

    mask = labels_flat != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)

    probas_flat = probas_flat[:, mask]  # (C, M)
    labels_flat = labels_flat[mask]  # (M,)

    present_classes = torch.unique(labels_flat)
    losses = []

    for c in present_classes:
        fg = (labels_flat == c).float()
        class_probas = probas_flat[c]

        errors = (fg - class_probas).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = lovasz_grad(fg_sorted)

        class_loss = torch.dot(errors_sorted, grad)
        losses.append(class_loss)

    return (
        torch.mean(torch.stack(losses))
        if losses
        else torch.tensor(0.0, device=preds.device)
    )
