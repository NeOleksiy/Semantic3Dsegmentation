import streamlit as st
import numpy as np
import tempfile
import torch
import plotly.graph_objs as go
from plyfile import PlyData
import open3d as o3d
import logging
from pointnet import PointNet

# Конфигурация логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы классов и цветов
COLOR_LEGEND_DATA = [
    (1, "wall", (174, 199, 232)),
    (2, "floor", (152, 223, 138)),
    (3, "cabinet", (31, 119, 180)),
    (4, "bed", (255, 187, 120)),
    (5, "chair", (188, 189, 34)),
    (6, "sofa", (140, 86, 75)),
    (7, "table", (255, 152, 150)),
    (8, "door", (214, 39, 40)),
    (9, "window", (197, 176, 213)),
    (10, "bookshelf", (148, 103, 189)),
    (11, "picture", (196, 156, 148)),
    (12, "counter", (23, 190, 207)),
    (14, "desk", (247, 182, 210)),
    (15, "curtain", (66, 188, 102)),
    (16, "refrigerator", (219, 219, 141)),
    (17, "shower curtain", (140, 57, 197)),
    (18, "toilet", (202, 185, 52)),
    (19, "sink", (51, 176, 203)),
    (20, "bathtub", (200, 54, 131)),
    (40, "otherfurniture", (100, 85, 144))
]

def show_color_legend():
    """Отображает цветовую легенду классов с помощью Streamlit-компонентов"""
    st.subheader("Цветовая легенда классов")
    
    # Создаем 3 колонки
    cols = st.columns(3)
    
    for idx, (color_id, label, color) in enumerate(COLOR_LEGEND_DATA):
        with cols[idx % 3]:
            st.markdown(
                f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                f"<div style='width: 30px; height: 30px; background-color: rgb{color}; "
                f"margin-right: 10px; border: 1px solid #ddd;'></div>"
                f"<div>{label} (ID: {color_id})</div>"
                f"</div>", 
                unsafe_allow_html=True
            )

def try_load_ply(file_path):
    """Загрузка PLY-файла с обработкой ошибок"""
    try:
        # Попытка чтения через plyfile
        plydata = PlyData.read(file_path)
        if 'vertex' in plydata:
            vertices = plydata['vertex']
            return np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    except Exception as e:
        logger.warning(f"Ошибка чтения через plyfile: {str(e)}")

    try:
        # Резервный метод через Open3D
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {str(e)}")
        return None

def load_and_preprocess(file_path, num_points=10000):
    """Предобработка данных"""
    try:
        points = try_load_ply(file_path)
        if points is None or len(points) == 0:
            raise ValueError("Не удалось загрузить точки")

        # Нормализация
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.linalg.norm(points, axis=1)) or 1.0
        points /= max_dist

        # Обрезка/дополнение точек
        if len(points) > num_points:
            points = points[np.random.choice(len(points), num_points, replace=False)]
        elif len(points) < num_points:
            points = np.vstack([points, np.zeros((num_points - len(points), 3))])

        return torch.tensor(points.T, dtype=torch.float32).unsqueeze(0)
    
    except Exception as e:
        st.error(f"Ошибка обработки данных: {str(e)}")
        return None

def visualize_point_cloud(points, labels=None, title="Point Cloud"):
    """Визуализация облака точек"""
    try:
        points = points.squeeze().numpy().T
        valid_mask = np.linalg.norm(points, axis=1) > 0
        valid_points = points[valid_mask]

        colors = []
        if labels is not None:
            valid_labels = labels[valid_mask]
            color_map = {id: color for id, _, color in COLOR_LEGEND_DATA}
            for lbl in valid_labels:
                color = color_map.get(lbl, (0, 0, 0))
                colors.append(f'rgb{color}')

        fig = go.Figure(data=[go.Scatter3d(
            x=valid_points[:,0],
            y=valid_points[:,1],
            z=valid_points[:,2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors if labels is not None else 'blue',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка визуализации: {str(e)}")

@st.cache_resource
def load_model(model_path='/home/efimenko.aleksey7/best_model/best_point_net.pt'):
    """Загрузка предобученной модели"""
    try:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        model = PointNet().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None

def main():
    st.title("3D Point Cloud Segmentation with PointNet")
    show_color_legend()

    uploaded_file = st.file_uploader("Загрузите PLY-файл", type="ply")
    if not uploaded_file:
        return

    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        input_tensor = load_and_preprocess(tmp_file.name)

    if input_tensor is None:
        return

    st.header("Исходное облако точек")
    visualize_point_cloud(input_tensor)

    model = load_model()
    if model is None:
        return

    if st.button("Выполнить сегментацию"):
        with st.spinner("Идет обработка..."):
            try:
                with torch.no_grad():
                    output = model(input_tensor)
                    labels = torch.argmax(output, dim=1).squeeze(0).numpy()

                st.header("Результат сегментации")
                visualize_point_cloud(input_tensor, labels)

            except Exception as e:
                st.error(f"Ошибка сегментации: {str(e)}")

if __name__ == "__main__":
    main()