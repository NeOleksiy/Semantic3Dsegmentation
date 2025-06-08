import os
import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from .utils import optimized_fps

def resample_points_fps(points, labels, target_size, allowed_labels=None):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    total_points = points.shape[0]
    device = torch.device('cpu')

    # identify rare classes
    if allowed_labels:
        hist = np.bincount(labels, minlength=len(self.allowed_labels))
    else:
        hist = np.bincount(labels)
        
    rare_mask = hist < target_size//100
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
        new_points = (class_points[pairs[:,0]] + class_points[pairs[:,1]]) / 2
        return np.vstack([class_points, new_points])[:target_size]

    resampled = []
    for label in unique_labels:
        mask = class_masks[label]
        class_points = points[mask]
        target = hist[label] if label in rare_labels else targets[frequent_labels.index(label)]
        
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
    final_labels = np.concatenate([np.full(len(arr), label) 
                                  for label, arr in zip(unique_labels, resampled)])
    
    shuffle_idx = np.random.permutation(len(final_points))
    return final_points[shuffle_idx][:target_size], final_labels[shuffle_idx][:target_size]

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