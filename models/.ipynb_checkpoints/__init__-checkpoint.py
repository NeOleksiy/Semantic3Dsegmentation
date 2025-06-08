import sys
from .pointnet import PointNet
from .pointnetplus import PointNetPlusPlus
from .voxelnet import VoxelNet
from .voxelnetplus import EnhancedVoxelNetPlus
from .dgcnn import DGCNN
sys.path.append("pvcnn")
from .external.pvcnn.models.s3dis import PVCNN2
import importlib

def model_factory(cfg):

    model_type = cfg.model.name
    
    if model_type == "PointNet":
        return PointNet(
            num_points=cfg.get("num_points", 10000),
            num_classes=cfg.get("num_classes", 20),
            num_global_features=cfg.get("num_global_features", 1024),
        )
    
    elif model_type == "PointNet++":
        return PointNetPlusPlus(
            num_classes=cfg.get("num_classes", 20),
            extra_feature_channels=cfg.get("extra_feature_channels", 3),
            dropout=cfg.get("dropout", 0.5),
        )
    
    elif model_type == "VoxelNet":
        return VoxelNet(
            num_classes=cfg.get("num_classes", 20),
            grid_size=cfg.get("grid_size", 64),
            dropout=cfg.get("dropout", 0.5),
        )
    
    elif model_type == "VoxelNet++":
        return EnhancedVoxelNetPlus(
            num_classes=cfg.get("num_classes", 20),
            grid_size=cfg.get("grid_size", 64),
            extra_feature_channels=cfg.get("extra_feature_channels", 3),
            dropout=cfg.get("dropout", 0.5),
        )

    elif model_type == "DGCNN":
        return DGCNN(
            num_classes=cfg.get("num_classes", 20),
            extra_feature_channels=cfg.get("extra_feature_channels", 3),
            k=cfg.get("k", 20),
            emb_dims=cfg.get("emb_dims", 1024),
            dropout=cfg.get("dropout", 0.5),
        )
        
    elif model_type == "PVCNN":
        return PVCNN2(
            num_classes=cfg.get("num_classes", 20),
            extra_feature_channels=cfg.get("extra_feature_channels", 3),
            width_multiplier=cfg.get("width_multiplier",1),
            voxel_resolution_multiplier=cfg.get("voxel_resolution_multiplier",1)
        ) 
    else:
        raise ValueError(f"Unknown type model: {model_type}")