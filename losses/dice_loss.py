import torch
import torch.nn as nn
import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(
#         self,
#         smooth: float = 1.0,
#         ignore_index: int = -100,
#         reduction: str = "mean",
#     ):

#         super().__init__()
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

#         num_classes = logits.shape[1]
#         probs = F.softmax(logits, dim=1)  # [B, C, D, H, W]

#         targets_onehot = (
#             F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
#             .permute(0, 4, 1, 2, 3)
#             .float()
#         )  # [B, C, D, H, W]

#         mask = (targets != self.ignore_index).unsqueeze(1)  # [B, 1, D, H, W]
#         probs = probs * mask
#         targets_onehot = targets_onehot * mask

#         intersection = (probs * targets_onehot).sum(dim=(2, 3, 4))  # [B, C]
#         union = probs.sum(dim=(2, 3, 4)) + targets_onehot.sum(dim=(2, 3, 4))  # [B, C]

#         dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
#         loss = 1.0 - dice  # [B, C]

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss  # [B, C]


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
