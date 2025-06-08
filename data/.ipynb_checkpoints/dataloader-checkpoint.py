import torch
from torch.utils.data import DataLoader
from .dataset import get_dataset

def get_dataloader(cfg, split='train'):
    """Create DataLoader"""
    dataset = get_dataset(cfg.data, split=split)
    
    if dataset.sample_type == "non_sample":
        collate_fn = collate_fn_non_sample
    else:
        collate_fn = None
    
    return DataLoader(
        dataset,
        batch_size=cfg.dataloader.train.batch_size if split == 'train' else cfg.dataloader.valid.batch_size,
        shuffle=(split == 'train'),
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

def collate_fn_non_sample(batch):
    features, labels = zip(*batch)
    
    max_points = max(f.shape[1] for f in features)
    if max_points > 1000000:
        max_points = 1000000
    
    padded_features = torch.zeros(len(features), 6, max_points)
    padded_labels = torch.zeros(len(features), max_points, dtype=torch.long)
    
    for i, (f, l) in enumerate(zip(features, labels)):
        current_points = f.shape[1]
        if current_points > max_points:
            idx = torch.randperm(current_points)[:max_points]
            padded_features[i] = f[:, idx]
            padded_labels[i] = l[idx]
        else:
            padded_features[i, :, :current_points] = f
            padded_labels[i, :current_points] = l
    
    return padded_features, padded_labels