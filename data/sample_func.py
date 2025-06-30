import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d
import torch
from sklearn.neighbors import NearestNeighbors

from .utils import optimized_fps


def resample_points_fps(points, labels, target_size, allowed_labels=None):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    total_points = points.shape[0]
    device = torch.device("cpu")

    # identify rare classes
    if allowed_labels is not None:
        hist = np.bincount(labels, minlength=len(allowed_labels))
    else:
        hist = np.bincount(labels)

    rare_mask = hist < target_size // 100
    rare_labels = np.where(rare_mask)[0]

    class_masks = {label: labels == label for label in unique_labels}

    remaining_target = target_size - hist[rare_labels].sum()
    frequent_labels = [l for l in unique_labels if l not in rare_labels]

    # exponential weighting
    if frequent_labels:
        freq_counts = hist[frequent_labels]
        weights = np.power(freq_counts, 0.75)
        targets = (remaining_target * weights / weights.sum()).astype(int)
        targets[-1] += remaining_target - targets.sum()

    # upsample
    def batch_upsample(class_points, target_size):
        n = len(class_points)
        if n >= target_size:
            return class_points

        pairs = np.random.choice(n, size=(target_size - n, 2), replace=True)
        new_points = (class_points[pairs[:, 0]] + class_points[pairs[:, 1]]) / 2
        return np.vstack([class_points, new_points])[:target_size]

    resampled = []
    for label in unique_labels:
        mask = class_masks[label]
        class_points = points[mask]
        target = (
            hist[label]
            if label in rare_labels
            else targets[frequent_labels.index(label)]
        )

        if len(class_points) == 0:
            continue

        if len(class_points) > target:
            fps_idx = optimized_fps(class_points, target)
            resampled.append(class_points[fps_idx])
        elif len(class_points) < target:
            resampled.append(batch_upsample(class_points, target))
        else:
            resampled.append(class_points)

    final_points = np.concatenate(resampled, axis=0)
    final_labels = np.concatenate(
        [np.full(len(arr), label) for label, arr in zip(unique_labels, resampled)]
    )

    shuffle_idx = np.random.permutation(len(final_points))
    return (
        final_points[shuffle_idx][:target_size],
        final_labels[shuffle_idx][:target_size],
    )


def resample_points(points, labels, target_size):
    current_size = points.shape[0]

    # Downsampling
    if current_size > target_size:
        indices = np.random.choice(current_size, target_size, replace=False)
        return points[indices], labels[indices]

    # Upsampling
    if current_size < target_size:
        nbrs = NearestNeighbors(n_neighbors=2).fit(points[:, :3])
        new_points = np.copy(points)
        new_labels = np.copy(labels)

        while len(new_points) < target_size:
            idx = np.random.choice(len(new_points))
            _, neighbors = nbrs.kneighbors([new_points[idx, :3]])
            neighbor_idx = neighbors[0, 1]

            new_point = (new_points[idx] + new_points[neighbor_idx]) / 2
            new_label = new_labels[idx]

            new_points = np.vstack([new_points, new_point])
            new_labels = np.append(new_labels, new_label)

        return new_points[:target_size], new_labels[:target_size]

    return points, labels


import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# def optimized_resample_points_fps(points, labels, target_size, k=50, batch_size=10000):
# import numpy as np
# from scipy.spatial import cKDTree
# from tqdm import tqdm


def optimized_resample_points_fps(points, labels, target_size, k=20):

    spatial_points = points[:, :3]
    n_points = len(points)

    if n_points <= target_size:
        nbrs = NearestNeighbors(n_neighbors=2).fit(points[:, :3])
        new_points = np.copy(points)
        new_labels = np.copy(labels)

        while len(new_points) < target_size:
            idx = np.random.choice(len(new_points))
            _, neighbors = nbrs.kneighbors([new_points[idx, :3]])
            neighbor_idx = neighbors[0, 1]

            new_point = (new_points[idx] + new_points[neighbor_idx]) / 2
            new_label = new_labels[idx]

            new_points = np.vstack([new_points, new_point])
            new_labels = np.append(new_labels, new_label)

        return new_points[:target_size], new_labels[:target_size]

    # Используем только первые 10000 точек для построения KDTree (ускорение в 10+ раз)
    sample_indices = np.random.choice(
        n_points, size=min(10000, n_points), replace=False
    )
    sample_points = spatial_points[sample_indices]
    tree = cKDTree(sample_points)

    # Быстрый расчет плотности на основе подвыборки
    dists, _ = tree.query(spatial_points, k=min(4, len(sample_points)), workers=-1)
    density = 1 / np.mean(dists[:, 1:], axis=1)

    # Упрощенный расчет граничных показателей
    if labels.ndim > 1 or not np.issubdtype(labels.dtype, np.integer):
        # Для цветов: случайная подвыборка для вычисления дисперсии
        sample_idx = np.random.choice(n_points, size=min(5000, n_points), replace=False)
        edge_scores = np.zeros(n_points)
        edge_scores[sample_idx] = np.var(labels[sample_idx], axis=0).mean()
    else:
        # Для семантики: простой подсчет доминирующего класса
        unique, counts = np.unique(labels, return_counts=True)
        dominant_class = unique[np.argmax(counts)]
        edge_scores = (labels != dominant_class).astype(float)

    # Комбинированная важность
    importance = 0.8 * density + 0.2 * edge_scores
    importance = np.nan_to_num(importance, nan=0.0)

    # Выборка топ-N точек по важности (быстрее вероятностной выборки)
    if np.sum(importance) > 0:
        probs = importance / np.sum(importance)
    else:
        probs = np.ones(n_points) / n_points

    selected_idx = np.random.choice(n_points, size=target_size, replace=False, p=probs)

    return points[selected_idx], labels[selected_idx]
