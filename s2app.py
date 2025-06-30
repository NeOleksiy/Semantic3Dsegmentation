import importlib.util
import logging
import sys
import tempfile
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objs as go
import streamlit as st
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data.dataset import ScannetDataset
from data.sample_func import resample_points
from inference import PointCloudSegmenter
from models import model_factory
from utils.constants import COLOR_MAP, LEGEND_DATA, convert_to_original_labels
from utils.logging import setup_logger
from utils.utils import compute_curvature, compute_density, save_pointcloud


def show_color_legend():
    st.subheader("Цветовая легенда классов")
    cols = st.columns(3)
    for idx, (color_id, label, color) in enumerate(LEGEND_DATA):
        with cols[idx % 3]:
            st.markdown(
                f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                f"<div style='width: 30px; height: 30px; background-color: rgb{color}; "
                f"margin-right: 10px; border: 1px solid #ddd;'></div>"
                f"<div>{label} (ID: {color_id})</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def try_load_ply(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        return points, colors, normals
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {str(e)}")
        return None, None, None


@st.cache_resource
def load_segmenter(_config, _logger):
    return PointCloudSegmenter(_config, _logger)


def visualize_point_cloud(points, colors=None, title="Point Cloud", max_points=12000):
    try:
        if points is None or len(points) == 0:
            st.warning("Облако точек пустое")
            return

        points = np.asarray(points, dtype=np.float32)

        if points.ndim == 2 and points.shape[0] == 3:
            points = points.T
        elif points.ndim != 2 or points.shape[1] != 3:
            st.error(f"Неподдерживаемая форма точек: {points.shape}")
            return

        n_points = len(points)

        centroid = np.mean(points, axis=0, dtype=np.float32)
        points_centered = points - centroid

        marker_color = None
        if colors is not None:
            colors = np.asarray(colors)

            if colors.dtype != np.uint8:
                if colors.max() > 1.0:
                    colors = np.clip(colors, 0, 255).astype(np.uint8)
                else:
                    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

            if colors.ndim == 1:
                marker_color = np.empty(n_points, dtype="U15")
                for i in range(n_points):
                    c = colors[i]
                    marker_color[i] = f"rgb({c},{c},{c})"
            elif colors.shape[1] >= 3:
                marker_color = np.empty(n_points, dtype="U20")
                rgb = colors[:, :3]
                for i in range(n_points):
                    r, g, b = rgb[i]
                    marker_color[i] = f"rgb({r},{g},{b})"

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points_centered[:, 0],
                    y=points_centered[:, 1],
                    z=points_centered[:, 2],
                    mode="markers",
                    marker=dict(
                        size=1,
                        color=(
                            marker_color
                            if marker_color is not None
                            else "rgb(70, 130, 180)"
                        ),
                        opacity=0.7,
                        line=dict(width=0),
                    ),
                    hoverinfo="none",
                )
            ],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=14)),
                scene=dict(
                    xaxis=dict(title="X", showspikes=False, showbackground=False),
                    yaxis=dict(title="Y", showspikes=False, showbackground=False),
                    zaxis=dict(title="Z", showspikes=False, showbackground=False),
                    aspectmode="data",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=500,
            ),
        )

        fig.update_layout(
            uirevision="constant",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as e:
        st.error(f"Ошибка визуализации: {str(e)}")
        if "points" in locals():
            st.write(f"Точек: {len(points)}")
        if "colors" in locals() and colors is not None:
            st.write(f"Цветов: {len(colors)}")


def main():
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path="configs")
    _logger = setup_logger()
    cfg = hydra.compose(config_name="sl_app")

    st.title("3D Point Cloud Segmentation")
    st.info(f"Using model: {cfg.model.name}")
    show_color_legend()

    segmenter = load_segmenter(cfg, _logger)

    uploaded_file = st.file_uploader("Загрузите PLY-файл", type="ply")
    if not uploaded_file:
        return

    with tempfile.NamedTemporaryFile(suffix=".ply") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        points, colors, normals = try_load_ply(tmp_file.name)

    if points is None:
        st.error("Не удалось загрузить файл")
        return

    st.sidebar.subheader("Информация о файле")
    st.sidebar.info(
        f"Точек: {len(points)}\nЦвета: {'да' if colors is not None else 'нет'}\nНормали: {'да' if normals is not None else 'нет'}"
    )

    st.header("Исходное облако точек")

    try:
        input_tensor = segmenter.preprocess(points, colors, normals)
        visualize_point_cloud(
            input_tensor[:, :3].cpu().numpy(), colors=input_tensor[:, 3:6].cpu().numpy()
        )
    except Exception as e:
        st.error(f"Ошибка предобработки: {str(e)}")
        return

    if st.button("Выполнить сегментацию"):
        with st.spinner("Идет обработка..."):
            try:

                predictions = segmenter.predict(input_tensor)
                pred_colors = segmenter.colorize(
                    input_tensor[:, :3].cpu().numpy(), predictions
                )

                st.header("Результат сегментации")

                visualize_point_cloud(
                    input_tensor[:, :3].cpu().numpy(), colors=pred_colors
                )

            except Exception as e:
                st.error(f"Ошибка сегментации: {str(e)}")


if __name__ == "__main__":
    main()
