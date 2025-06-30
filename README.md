# 3D Semantic Segmentation Toolkit

![3D Segmentation Example](sem_3d_seg/static/seg_screan.png)  
*Semantic segmentation example on ScanNet scene*

## Project Overview

This repository provides a comprehensive framework for training and deploying 3D semantic segmentation models. The implementation supports various neural architectures for ScanNet dataset with different data representational.


### Dependencies
```bash
# Python 3.8+
make setup

```

# Dataset Setup
1.`
make data-download
`
2.Organize data with this structure:
```
datasets/
└── scannet/
    ├── scenes0000_00/
    │   ├── scene0000_00_vh_clean_2.ply (raw points cloud scene)
    |   |── scene0000_00_vh_clean_2.labels.ply (labels points cloud scene)
    |   |── scene0000_00_vh_clean_2.0.010000.segs.json
    |   |── scene0000_00.txt (scene information)
    |   |── scene0000_00.aggregation.json (aggregation per classes)
    |   |──────
    ├── scenes0000_01/
    │   ├── scene0000_01_vh_clean_2.ply
    |   |── scene0000_01_vh_clean_2.labels.ply
    |   |── scene0000_01_vh_clean_2.0.010000.segs.json
    |   |── scene0000_01.txt
    |   |── scene0000_01.aggregation.json
    |   |──────
...



# Training
```python
make train 
```
# Inference
```python
make infer input_path=scene0022_01_vh_clean_2.ply \ #need fill inference.yaml
                        output_path=results/segmented.ply
```
# StreamLit visualization
```python
make streamlit_app.py #need fill sl_app.yaml
```

# Docker
```bash
make docker-build
make docker-run-train # for train
make docker-run-infer input_path=scene0022_01_vh_clean_2.ply output_path=/workspace/results/segmented.ply # for inference
```

# Repository Structure

3d-seg-repo/
├── configs/             # configuration
├── data/                # Data modules(dataset, augmentation)
├── models/              # Model implementations
├── utils/               # utils(and losses)
├── evaluation/          # Evaluation metrics
├── losses/              # Losses class 
├── docker/              # Docker container
├── static/              # static file
├── inference.py         # Inference script
├── train.py             # Main training script
├── s2app.py             # streamlit_app
├── requirements.txt     # Python dependencies
├── Makefile

## Supported Models

| Model            | Support Extra Features<br>(RGB, Normals) | mIoU<br>(ScanNet sample) | mIoU<br>(KITTI360 )     |
|------------------|------------------------------------------|--------------------------|--------------------------|
| PointNet         | ❌                                       | 21.6%                    | -                        |
| PointNet++       | ✅                                       | 39.4%                    | 18.6%                    |
| VoxelNet         | ❌                                       | 19.2%                    | -                        |
| VoxelNet+        | ✅                                       | 20.7%                    | 8.2%                     |
| DGCNN            | ✅                                       | 30.1%                    | 21.6%                    |
| PVCNN            | ✅                                       | 46.1%                    | 21.6%                    |

### PVCNN:
```
@inproceedings{liu2019pvcnn,
  title={Point-Voxel CNN for Efficient 3D Deep Learning},
  author={Liu, Zhijian and Tang, Haotian and Lin, Yujun and Han, Song},
  booktitle={NeurIPS},
  year={2019}
}
```