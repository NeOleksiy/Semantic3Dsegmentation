import argparse
import logging
import os
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from omegaconf import DictConfig, OmegaConf

from data.dataset import ScannetDataset
from data.sample_func import resample_points
from models import model_factory
from utils.constants import COLOR_MAP, convert_to_original_labels
from utils.logging import setup_logger
from utils.utils import compute_curvature, compute_density, save_pointcloud


class PointCloudSegmenter:
    """End-to-end point cloud segmentation pipeline"""

    def __init__(self, config, logger):
        self.config = config
        self.device = self._setup_device()
        self.model = self._load_model()
        self.logger = logger
        self.logger.info(f"Model loaded: {config['model']['name']} on {self.device}")

    def _setup_device(self):
        """Select appropriate device based on availability"""
        if self.config["device"] == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config["device"])

    def _load_model(self):
        """Load model from checkpoint"""
        from models import model_factory

        model = model_factory(self.config).to(self.device)

        # Load weights
        if not os.path.exists(self.config["model_path"]):
            raise FileNotFoundError(f"Model not found: {self.config['model_path']}")

        checkpoint = torch.load(self.config["model_path"], map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    def load_point_cloud(self, file_path):
        """Load and validate point cloud"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError("Point cloud contains no points")

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
        normals = (
            np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
        )

        self.logger.info(f"Loaded point cloud: {len(points)} points")
        return points, colors, normals

    def preprocess(self, points, colors, normals):
        """Prepare input features based on model type"""
        # Create feature vector
        self.logger.info("Start preprocess points and features")
        features = [points]

        if self.config["use_rgb"] and colors.any():
            features.append(colors)

        if self.config["use_normals"] and normals.any():
            features.append(normals)

        if self.config["make_curvature"]:
            features.append(compute_curvature(points))

        if self.config["make_density"]:
            features.append(compute_density(points))

        features = np.hstack(features)
        self.logger.info(
            f"End preprocess points and features, total shape: {features.shape}"
        )
        features, _ = resample_points(features, features, self.config["target_size"])
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def predict(self, input_data):
        """Run model inference"""
        start_time = time.time()
        self.logger.info("Start prediction")
        # Batch processing

        with torch.no_grad():
            output = self.model(input_data.permute(1, 0).unsqueeze(0))
            preds = torch.argmax(output, dim=1)
            predictions = preds.cpu().numpy()

        self.logger.info(f"Inference completed in {time.time()-start_time:.2f}s")
        return predictions.squeeze(0)

    def colorize(self, points, predictions):
        """Apply color mapping to predictions"""
        from evaluation.visualize import colorize_pointcloud

        pred_colors = np.array(
            [
                COLOR_MAP[m]
                for m in convert_to_original_labels(torch.tensor(predictions))
            ]
        )
        return pred_colors / 255.0

    def save_results(self, points, colors, output_path):
        """Save colored point cloud"""

        save_pointcloud(points, colors, output_path)
        self.logger.info(f"Results saved to {output_path}")

    def process(self, input_path, output_path):
        """End-to-end processing of a single PLY file"""
        try:
            # Load and process
            points, colors, normals = self.load_point_cloud(input_path)
            input_data = self.preprocess(points, colors, normals)
            predictions = self.predict(input_data)
            colored_points = self.colorize(input_data[:, :3].cpu().numpy(), predictions)

            # Save results
            self.save_results(
                input_data[:, :3].cpu().numpy(), colored_points, output_path
            )

            # Visualize if requested
            if self.config["visualize"]:
                self.visualize(input_data[:, :3].cpu().numpy(), colored_points)

            return True
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return False

    def visualize(self, points, colors):
        """Interactive visualization"""
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=1)
        plt.savefig("results/mask.png")
        plt.show()


@hydra.main(config_path="./configs", config_name="inference", version_base="1.2")
def main(cfg: DictConfig) -> None:

    logger = setup_logger()

    # Validate input
    if not Path(cfg.input_path).exists():
        logger.error(f"Input file not found: {cfg.input_path}")
        return

    # Create output directory
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("Starting 3D Point Cloud Segmentation")
    logger.info(f"Input:  {cfg.input_path}")
    logger.info(f"Output: {cfg.output_path}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Model:  {cfg.model.name}")

    # Process the file
    segmenter = PointCloudSegmenter(cfg, logger)
    success = segmenter.process(cfg.input_path, cfg.output_path)

    if success:
        logger.info("Processing completed successfully!")
    else:
        logger.error("Processing failed")


if __name__ == "__main__":
    main()
