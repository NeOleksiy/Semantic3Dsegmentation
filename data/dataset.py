import os

import numpy as np
import open3d as o3d  # type: ignore
import torch
from plyfile import PlyData
from sklearn.neighbors import KDTree, NearestNeighbors
from torch.utils.data import Dataset

from .augmentation import test_transform, train_transform
from .sample_func import (region_aware_sampling, resample_points,
                          resample_points_fps)


class ScannetDataset(Dataset):
    def __init__(
        self,
        root="downloaded/scans",
        is_train=True,
        transform=None,
        train_size=0.9,
        target_size=10000,
        rgb_flag=True,
        normal_flag=False,
        density_flag=False,
        curvature_flag=False,
        sample_type="sample",
        density_radius=0.1,
        name="Scannet",
    ):
        self.root = root
        self.files = os.listdir(root)
        self.rgb_flag = rgb_flag
        self.normal_flag = normal_flag
        self.density_flag = density_flag
        self.curvature_flag = curvature_flag
        self.sample_type = sample_type
        self.density_radius = density_radius
        self.transform = transform
        self.target_size = target_size
        if normal_flag:
            for scene in self.files:
                normals_path = os.path.join(self.root, scene, f"{scene}_normals.npy")
                if not os.path.exists(normals_path):
                    raise FileNotFoundError(
                        f"Normals not found for {scene}. Run precompute_scannet_normals() first."
                    )

        # Split train/test
        split_idx = int(train_size * len(self.files))
        self.files = self.files[:split_idx] if is_train else self.files[split_idx:]

        # Label mapping
        self.allowed_labels = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.allowed_labels)
        }
        self.idx_to_label = {
            idx: label for idx, label in enumerate(self.allowed_labels)
        }

    def __len__(self):
        return len(self.files)

    def _load_ply(self, path, scene_id):
        plydata = PlyData.read(path)
        vertex = plydata["vertex"]

        # Load coordinates
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

        # Load RGB
        if self.rgb_flag:
            rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
            rgb = rgb.astype(np.float32) / 255.0
            points = np.hstack([points, rgb])

        # Load normals
        if self.normal_flag:
            normals_path = os.path.join(
                os.path.dirname(path), f"{scene_id}_normals.npy"
            )
            normals = np.load(normals_path)
            points = np.hstack([points, normals])

        return points

    def _load_labels(self, path):
        plydata = PlyData.read(path)
        labels = np.array(plydata["vertex"]["label"])
        labels = np.where(np.isin(labels, self.allowed_labels), labels, 39)
        labels = np.vectorize(self.label_to_idx.get)(labels)
        return labels

    def _compute_density(self, coords):
        tree = KDTree(coords)
        densities = np.array(
            [len(tree.query_radius([pt], r=self.density_radius)[0]) for pt in coords]
        )
        return densities.reshape(-1, 1)

    def _compute_curvature(self, coords, k=30):
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

    def convert_to_original_labels(self, indices_tensor):
        if isinstance(indices_tensor, torch.Tensor):
            indices = indices_tensor.cpu().numpy()
        return np.vectorize(self.idx_to_label.get)(indices)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        ply_path = os.path.join(path, f"{os.path.basename(path)}_vh_clean_2.labels.ply")
        ply_clean_path = os.path.join(path, f"{os.path.basename(path)}_vh_clean_2.ply")

        # Load data
        points = self._load_ply(ply_clean_path, self.files[idx])
        labels = self._load_labels(ply_path)

        if self.sample_type == "fps_sample":
            points, labels = resample_points_fps(
                points, labels, self.target_size, self.allowed_labels
            )
        if self.sample_type == "sample":
            points, labels = resample_points(points, labels, self.target_size)

        if self.sample_type == "region_aware":
            points, labels = region_aware_sampling(
                points, labels, self.target_size
            )

        # Compute additional features
        coords = points[:, :3]
        if self.density_flag:
            densities = self._compute_density(coords)
            points = np.hstack([points, densities])

        if self.curvature_flag:
            curvatures = self._compute_curvature(coords)
            points = np.hstack([points, curvatures])

        data_dict = {
            "pos": points.astype(np.float32),
            "labels": labels.astype(np.int32),
        }
        # augmentation
        if self.transform:
            data_dict = self.transform(data_dict)

        pos = data_dict["pos"]
        labels = data_dict["labels"]

        if isinstance(pos, torch.Tensor):
            points_tensor = pos.T.float()  # (num_features, num_points)
            labels_tensor = labels.long()  # (num_points,)
        else:
            points_tensor = torch.from_numpy(pos).float().T
            labels_tensor = torch.from_numpy(labels).long()

        return points_tensor, labels_tensor


class KITTI360Dataset(Dataset):
    def __init__(
        self,
        root="downloaded",
        split_file="data_3d_semantics/train/2013_05_28_drive_train.txt",
        is_train=True,
        transform=None,
        target_size=300000,
        rgb_flag=True,
        normal_flag=False,
        density_flag=False,
        curvature_flag=False,
        sample_type="sample",
        density_radius=0.3,
        name="KITTI360",
    ):
        self.root = root
        self.rgb_flag = rgb_flag
        self.normal_flag = normal_flag
        self.density_flag = density_flag
        self.curvature_flag = curvature_flag
        self.sample_type = sample_type
        self.density_radius = density_radius
        self.transform = transform
        self.target_size = target_size

        with open(os.path.join(root, split_file), "r") as f:
            self.files = [line.strip() for line in f.readlines()]

        train_size = 0.9
        split_idx = int(train_size * len(self.files))
        self.files = self.files[:split_idx] if is_train else self.files[split_idx:]

        self.label_map = np.full((300,), 19, dtype=np.uint8)
        valid_labels = {
            0: 0,  # дорога
            10: 2,  # Здание
            8: 3,  # Стена
            9: 4,  # Забор
            17: 5,  # Столб
            19: 6,  # Светофор
            20: 7,  # Дорожный знак
            21: 8,  # Растительность
            22: 9,  # Природа
            24: 11,  # Человек
            26: 13,  # Автомобиль
            27: 14,  # Грузовик
            28: 12,  # Автобус
            32: 10,  # Мотоцикл
            33: 1,  # Велосипед
        }
        # valid_labels = {
        #     0: 19,   # Неизвестный
        #     1: 0,    # Дорога
        #     2: 1,    # Тротуар
        #     10: 2,   # Здание
        #     8: 3,    # Стена
        #     9: 4,    # Забор
        #     17: 5,   # Столб
        #     19: 6,   # Светофор
        #     20: 7,   # Дорожный знак
        #     21: 8,   # Растительность
        #     22: 9,   # Природа
        #     23: 10,  # Небо
        #     24: 11,  # Человек
        #     25: 12,  # Всадник
        #     26: 13,  # Автомобиль
        #     27: 14,  # Грузовик
        #     28: 15,  # Автобус
        #     31: 16,  # Поезд
        #     32: 17,  # Мотоцикл
        #     33: 18,  # Велосипед
        # }

        self.allowed_labels = valid_labels.keys()

        for orig_id, mapped_id in valid_labels.items():
            self.label_map[orig_id] = mapped_id
        self.num_classes = 15

        if normal_flag:
            for file in self.files:
                ply_path = os.path.join(self.root, file)
                norm_path = ply_path.replace(".ply", "_normals.npy")
                if not os.path.exists(norm_path):
                    self._precompute_normals(ply_path, norm_path)

    def __len__(self):
        return len(self.files)

    def _precompute_normals(self, ply_path, norm_path):
        """Вычисление и сохранение нормалей с использованием Open3D"""
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        normals = np.asarray(pcd.normals)
        np.save(norm_path, normals.astype(np.float32))

    def _load_ply(self, ply_path):
        """Загрузка облака точек и меток из PLY"""
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

        if self.rgb_flag:
            rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
            rgb = rgb.astype(np.float32) / 255.0
            points = np.hstack([points, rgb])

        if self.normal_flag:
            norm_path = ply_path.replace(".ply", "_normals.npy")
            normals = np.load(norm_path)
            points = np.hstack([points, normals])

        semantic = np.array(vertex["semantic"])
        labels = self.label_map[semantic]

        return points, labels

    def _compute_density(self, coords):
        """Расчет локальной плотности точек"""
        tree = KDTree(coords)
        densities = tree.query_radius(coords, r=self.density_radius, count_only=True)
        return np.array(densities).reshape(-1, 1)

    def _compute_curvature(self, coords, k=50):
        """Оценка кривизны через анализ соседей"""
        tree = KDTree(coords)
        curvatures = []
        for pt in coords:
            _, idxs = tree.query([pt], k=k)
            neighbors = coords[idxs[0]]
            cov = np.cov(neighbors.T)
            eigvals = np.linalg.eigvalsh(cov)
            curvatures.append(eigvals[0] / (np.sum(eigvals) + 1e-8))
        return np.array(curvatures).reshape(-1, 1)

    def __getitem__(self, idx):
        ply_path = os.path.join(self.root, self.files[idx])

        points, labels = self._load_ply(ply_path)
        coords = points[:, :3]

        if self.density_flag:
            densities = self._compute_density(coords)
            points = np.hstack([points, densities])

        if self.curvature_flag:
            curvatures = self._compute_curvature(coords)
            points = np.hstack([points, curvatures])

        if self.sample_type == "fps_sample":
            points, labels = resample_points_fps(
                points, labels, self.target_size, self.allowed_labels
            )
        if self.sample_type == "sample":
            points, labels = resample_points(points, labels, self.target_size)

        if self.sample_type == "region_aware":
            points, labels = region_aware_sampling(
                points, labels, self.target_size
            )

        data = {"pos": points.astype(np.float32), "labels": labels.astype(np.int64)}

        if self.transform:
            data = self.transform(data)

        pos = data["pos"]
        labels = data["labels"]

        return pos.T, labels


def get_dataset(cfg, split="train"):
    """Dataset Factory"""
    dataset_name = cfg.name.lower()

    if dataset_name == "scannet":
        return ScannetDataset(
            is_train=True if split == "train" else False,
            transform=train_transform if split == "train" else test_transform,
            **cfg,
        )
    if cfg.name.lower() == "kitti360":
        return KITTI360Dataset(
            root=cfg.root,
            split_file=cfg.split_file,
            is_train=(split == "train"),
            transform=train_transform if split == "train" else test_transform,
            target_size=cfg.target_size,
            rgb_flag=cfg.rgb_flag,
            normal_flag=cfg.normal_flag,
            density_flag=cfg.density_flag,
            curvature_flag=cfg.curvature_flag,
            sample_type=cfg.sample_type,
            density_radius=cfg.density_radius,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")
