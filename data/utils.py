import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

CLASS_LABELS = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)

SCANNET_COLOR_MAP = {
    0: (0, 0, 0),          # unlabeled
    1: (174, 199, 232),    # wall
    2: (152, 223, 138),    # floor
    3: (31, 119, 180),     # cabinet
    4: (255, 187, 120),    # bed
    5: (188, 189, 34),     # chair
    6: (140, 86, 75),      # sofa
    7: (255, 152, 150),    # table
    8: (214, 39, 40),      # door
    9: (197, 176, 213),    # window
    10: (148, 103, 189),   # bookshelf
    11: (196, 156, 148),   # picture
    12: (23, 190, 207),    # counter
    14: (247, 182, 210),   # desk
    16: (219, 219, 141),   # curtain
    24: (255, 127, 14),    # refrigerator
    28: (158, 218, 229),   # shower curtain
    33: (44, 160, 44),     # toilet
    34: (112, 128, 144),   # sink
    36: (227, 119, 194),   # bathtub
    39: (82, 84, 163)      # otherfurn
}

def precompute_scannet_normals(root_dir, overwrite=False):
    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for scene in tqdm(scenes):
        scene_dir = os.path.join(root_dir, scene)
        ply_path = os.path.join(scene_dir, f"{scene}_vh_clean_2.ply")
        normals_path = os.path.join(scene_dir, f"{scene}_normals.npy")
        
        if not overwrite and os.path.exists(normals_path):
            continue
            
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # compute normal
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        normals = np.asarray(pcd.normals)
        
        np.save(normals_path, normals.astype(np.float32))

def optimized_fps(points, k):
    n = len(points)
    if n <= k:
        return np.arange(n)
        
    points_tensor = torch.from_numpy(points[:, :3]).float()
    indices = torch.zeros(k, dtype=torch.long)
    
    distances = torch.norm(points_tensor - points_tensor[0], dim=1)
    indices[0] = 0
    indices[1] = torch.argmax(distances)
    
    for i in range(2, k):
        new_dist = torch.norm(points_tensor - points_tensor[indices[i-1]], dim=1)
        distances = torch.minimum(distances, new_dist)
        indices[i] = torch.argmax(distances)
    
    return indices.numpy()

def calculate_alpha_balanced(dataset, beta=0.8, smooth=1e-6, max_alpha=5.0, device="cpu"):
    """
    Параметры:
        dataset (Tensor или Dataset): 
            - Если Tensor: shape [M], каждый элемент — кортеж (points, label)
            - Если Dataset: объект с методом __getitem__
        beta (float): регулирует баланс (0.5-1.0)
        smooth (float): численная стабильность
        max_alpha (float): ограничение весов редких классов
        device: устройство для выходного тензора

    Возвращает:
        alpha (Tensor): веса классов [num_classes]
    """
    all_labels = []
    
    if isinstance(dataset, torch.Tensor):
        iterator = dataset
    else:
        iterator = [dataset[i] for i in range(len(dataset))]
    
    for item in tqdm(iterator, desc="Collecting tags"):
        _, label = item
        all_labels.append(label.to(device))
    
    all_labels = torch.cat(all_labels).flatten()
    
    labels_np = all_labels.cpu().numpy()
    classes, counts = np.unique(labels_np, return_counts=True)
    freq = counts / counts.sum()
    
    adjusted_freq = freq ** beta
    alpha = 1.0 / (adjusted_freq + smooth)
    
    alpha = alpha / alpha.sum() * len(classes)
    alpha = np.clip(alpha, a_min=1.0, a_max=max_alpha)
    
    return torch.tensor(alpha, dtype=torch.float32, device=device)