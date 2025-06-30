import os

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


def compute_normals(root_dir, overwrite=False):
    scenes = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]

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
        new_dist = torch.norm(points_tensor - points_tensor[indices[i - 1]], dim=1)
        distances = torch.minimum(distances, new_dist)
        indices[i] = torch.argmax(distances)

    return indices.numpy()


def compute_density(coords):
    tree = KDTree(coords)
    densities = np.array(
        [len(tree.query_radius([pt], r=self.density_radius)[0]) for pt in coords]
    )
    return densities.reshape(-1, 1)


def compute_curvature(coords, k=30):
    tree = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(coords)
    curvatures = []
    for pt in coords:
        _, indices = tree.kneighbors([pt])
        neighbors = coords[indices[0]]
        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-8)
        curvatures.append(curvature)
    return np.array(curvatures).reshape(-1, 1)


def save_pointcloud(points: np.ndarray, colors: np.ndarray, file_path: str) -> None:
    """Save point cloud to PLY file with validation"""
    if points.shape[0] == 0:
        raise ValueError("Cannot save empty point cloud")

    if colors is not None and points.shape[0] != colors.shape[0]:
        raise ValueError("Points and colors must have same length")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(file_path, pcd)


def calculate_alpha_balanced(
    dataset, beta=0.8, smooth=1e-6, max_alpha=5.0, device="cpu"
):
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

    adjusted_freq = freq**beta
    alpha = 1.0 / (adjusted_freq + smooth)

    alpha = alpha / alpha.sum() * len(classes)
    alpha = np.clip(alpha, a_min=1.0, a_max=max_alpha)

    return torch.tensor(alpha, dtype=torch.float32, device=device)
