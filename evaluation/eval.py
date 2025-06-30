from torch.utils.data import DataLoader, Dataset

from data.augmentation import test_transform, train_transform
from data.dataset import KITTI360Dataset, ScannetDataset

train_dataset = KITTI360Dataset(transform=train_transform)
points, mask = train_dataset[3]
print(mask.shape)
print(points.shape)

import numpy as np
import torch
from tqdm import tqdm

from models.pvcnn_plus import PVCNN2

NUM_CLASSES = 20
# Инициализация модели и загрузка весов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("outputs2/saved_model/best.pt", weights_only=False)
model = PVCNN2(num_classes=NUM_CLASSES, extra_feature_channels=3).to(
    device
)  # PointNetPlusPlusSeg(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Инициализация DataLoader
test_dataset = ScannetDataset(
    is_train=False, transform=test_transform, sample_type="optimized_fps"
)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=10)

# Подготовка для вычисления метрик
total_correct = 0
total_points = 0
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))


def calculate_iou(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    iou = intersection / (
        union + 1e-10
    )  # Добавляем небольшое значение для избежания деления на 0
    return iou


with torch.no_grad():
    for feature, mask in tqdm(test_loader, desc="Validation"):
        feature, label = feature.to(device), mask.to(device)

        outputs = model(feature)

        # Преобразуем выходы в предсказания
        preds = torch.argmax(outputs, dim=1)  # размерность (batch, num_points)

        # Обновляем метрики
        total_correct += (preds == label).sum().item()
        total_points += label.numel()

        # Обновляем матрицу ошибок для mIoU
        mask = label < NUM_CLASSES  # Игнорируем точки с недопустимыми классами
        labels_filtered = label[mask]
        preds_filtered = preds[mask]

        for lt, lp in zip(labels_filtered.view(-1), preds_filtered.view(-1)):
            confusion_matrix[lt.item(), lp.item()] += 1

# Вычисляем метрики
accuracy = total_correct / total_points
print(f"Overall Accuracy: {accuracy:.4f}")

iou_per_class = calculate_iou(confusion_matrix)
miou = np.nanmean(iou_per_class)
print(f"Mean IoU: {miou:.4f}")

# Выводим IoU для каждого класса
for class_idx, iou in enumerate(iou_per_class):
    print(f"Class {class_idx} IoU: {iou:.4f}")
