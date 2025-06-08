import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from torchvision import transforms

class PointCloudToTensor:
    def __call__(self, data_dict):
        if not isinstance(data_dict['pos'], torch.Tensor):
            data_dict['pos'] = torch.tensor(data_dict['pos'], dtype=torch.float32)
        
        if not isinstance(data_dict['labels'], torch.Tensor):
            data_dict['labels'] = torch.tensor(data_dict['labels'], dtype=torch.long)
            
        return data_dict

class PointCloudNormalize:
    def __call__(self, data_dict):
        pos = data_dict['pos']
        
        if isinstance(pos, np.ndarray):
            mean = np.mean(pos[:, :3], axis=0)
            pos[:, :3] -= mean
            max_dist = np.max(np.linalg.norm(pos[:, :3], axis=1))
            pos[:, :3] /= (max_dist + 1e-8)
        else:
            mean = torch.mean(pos[:, :3], dim=0)
            pos[:, :3] -= mean
            max_dist = torch.max(torch.norm(pos[:, :3], dim=1))
            pos[:, :3] /= (max_dist + 1e-8)
            
        data_dict['pos'] = pos
        return data_dict

class PointCloudTranslate:
    """Сдвиг только координатной части"""
    def __init__(self, shift_range=[0.2, 0.2, 0.2]):
        self.shift_range = np.array(shift_range)
        
    def __call__(self, data_dict):
        pos = data_dict['pos']
        if pos.shape[1] > 3:
            coords, colors = pos[:, :3], pos[:, 3:]
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            coords += shift
            pos = np.hstack([coords, colors])
        else:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            pos += shift
        data_dict['pos'] = pos
        return data_dict

class PointCloudScale:
    """Масштабирование только координат"""
    def __init__(self, scale_range=[0.8, 1.2]):
        self.scale_range = scale_range
        
    def __call__(self, data_dict):
        pos = data_dict['pos']
        if pos.shape[1] > 3:
            coords, colors = pos[:, :3], pos[:, 3:]
            scale = np.random.uniform(*self.scale_range)
            coords *= scale
            pos = np.hstack([coords, colors])
        else:
            scale = np.random.uniform(*self.scale_range)
            pos *= scale
        data_dict['pos'] = pos
        return data_dict

class PointCloudJitter:
    """Шум только для координат"""
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
        
    def __call__(self, data_dict):
        pos = data_dict['pos']
        if pos.shape[1] > 3:
            coords, colors = pos[:, :3], pos[:, 3:]
            noise = np.clip(self.sigma * np.random.randn(*coords.shape), -self.clip, self.clip)
            coords += noise
            pos = np.hstack([coords, colors])
        else:
            noise = np.clip(self.sigma * np.random.randn(*pos.shape), -self.clip, self.clip)
            pos += noise
        data_dict['pos'] = pos
        return data_dict

    
train_transform = transforms.Compose([
    PointCloudScale(),        # numpy 
    PointCloudTranslate(),    # numpy
    PointCloudJitter(),       # numpy
    PointCloudToTensor(),     # convert
    PointCloudNormalize(),    # tensor
])

test_transform = transforms.Compose([
    PointCloudToTensor(),     # convert
    PointCloudNormalize()    # tensor
])