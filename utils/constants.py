import numpy as np
import torch

allowed_labels = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)
label_to_idx = {label: idx for idx, label in enumerate(allowed_labels)}
idx_to_label = {idx: label for idx, label in enumerate(allowed_labels)}

CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)


def convert_to_original_labels(indices_tensor):
    if isinstance(indices_tensor, torch.Tensor):
        indices = indices_tensor.cpu().numpy()
    return np.vectorize(idx_to_label.get)(indices)


COLOR_MAP = {
    0: (0, 0, 0),  # unlabeled
    1: (174, 199, 232),  # wall
    2: (152, 223, 138),  # floor
    3: (31, 119, 180),  # cabinet
    4: (255, 187, 120),  # bed
    5: (188, 189, 34),  # chair
    6: (140, 86, 75),  # sofa
    7: (255, 152, 150),  # table
    8: (214, 39, 40),  # door
    9: (197, 176, 213),  # window
    10: (148, 103, 189),  # bookshelf
    11: (196, 156, 148),  # picture
    12: (23, 190, 207),  # counter
    14: (247, 182, 210),  # desk
    16: (219, 219, 141),  # curtain
    24: (255, 127, 14),  # refrigerator
    28: (158, 218, 229),  # shower curtain
    33: (44, 160, 44),  # toilet
    34: (112, 128, 144),  # sink
    36: (227, 119, 194),  # bathtub
    39: (82, 84, 163),  # otherfurn
}


LEGEND_DATA = [
    (1, "wall", (174, 199, 232)),
    (2, "floor", (152, 223, 138)),
    (3, "cabinet", (31, 119, 180)),
    (4, "bed", (255, 187, 120)),
    (5, "chair", (188, 189, 34)),
    (6, "sofa", (140, 86, 75)),
    (7, "table", (255, 152, 150)),
    (8, "door", (214, 39, 40)),
    (9, "window", (197, 176, 213)),
    (10, "bookshelf", (148, 103, 189)),
    (11, "picture", (196, 156, 148)),
    (12, "counter", (23, 190, 207)),
    (14, "desk", (247, 182, 210)),
    (15, "curtain", (66, 188, 102)),
    (16, "refrigerator", (219, 219, 141)),
    (17, "shower curtain", (140, 57, 197)),
    (18, "toilet", (202, 185, 52)),
    (19, "sink", (51, 176, 203)),
    (20, "bathtub", (200, 54, 131)),
    (40, "otherfurniture", (100, 85, 144)),
]
