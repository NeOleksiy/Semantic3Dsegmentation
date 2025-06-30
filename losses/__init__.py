import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data.dataset import get_dataset
from utils.utils import calculate_alpha_balanced

init_dataset_conf = OmegaConf.create(
    {
        "name": "ScanNet",
        "root": "../scans",
        "train_size": 0.9,
        "target_size": 10000,
        "sample_type": "sample",
        "rgb_flag": False,
        "normal_flag": False,
        "density_flag": False,
        "density_radius": 0.1,
        "curvature_flag": False,
    }
)


def get_loss(config: Dict[str, Any], logger: logging.Logger) -> nn.Module:

    loss_name = config["training"]["loss"]["name"].lower()
    loss_params = config["training"]["loss"].get("params", {})
    logger.info(f"Init {loss_name} loss")
    if loss_params.get("weight", False):
        logger.info(f"Calculate weight for classes of dataset")
        weight = calculate_alpha_balanced(get_dataset(init_dataset_conf, split="train"))
    else:
        weight = None

    if loss_name == "crossentropy":
        loss_fn = nn.CrossEntropyLoss(weight=weight)

    elif loss_name == "focal":
        from .focal_loss import FocalLoss

        alpha = loss_params.get("alpha", 0.5)
        gamma = loss_params.get("gamma", 2.0)
        reduct = loss_params.get("reduction", "mean")
        if weight is not None:
            loss_fn = FocalLoss(gamma=gamma, alpha=weight, reduction=reduct)
        loss_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduct)

    elif loss_name == "dice":
        from .dice_loss import DiceLoss

        smooth = loss_params.get("smooth", 1.0)
        reduct = loss_params.get("reduction", "mean")
        loss_fn = DiceLoss(smooth=smooth, reduction=reduct)

    elif loss_name == "lovasz":
        from .lovasz_loss import lovasz_softmax

        loss_fn = lovasz_softmax

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    return loss_fn
