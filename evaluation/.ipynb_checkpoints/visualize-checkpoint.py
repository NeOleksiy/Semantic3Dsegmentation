import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

def plot_segmentation_metrics(train_loss, val_loss, train_mean_iou=None, val_mean_iou=None, 
                              train_dice_score=None, val_dice_score=None):

    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plotting the Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plotting the Mean IoU (if available)
    if val_mean_iou:
        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_mean_iou, 'b', label='Test Mean IoU')
        plt.title('Test Mean IoU')
        plt.xlabel('Epochs')
        plt.ylabel('Mean IoU')
        plt.legend()
    
    # Plotting the Dice Score (if available)
    if train_dice_score and val_dice_score:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_dice_score, 'r', label='Training Dice Score')
        plt.plot(epochs, val_dice_score, 'b', label='Test Dice Score')
        plt.title('Training and Test Dice Score')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/losses.png')
    plt.show()

def visualize_2d_by_ply_path(path):
    plydata = PlyData.read(ply_file)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T 
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T / 255.0
    pcd_2d = points[:, :2] 

    plt.figure(figsize=(8, 8))
    plt.scatter(pcd_2d[:, 0], pcd_2d[:, 1], c=colors, s=1)
    plt.title('2D Projection of Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

def plot_2d_projections_voxel(voxel_grid, batch_idx=0):
    voxels = voxel_grid[batch_idx].cpu().detach().numpy()  # (C, D, H, W)
    
    aggregated = np.max(voxels, axis=0)  # (D, H, W)

    xy_proj = np.max(aggregated, axis=0)  # (H, W)
    xz_proj = np.max(aggregated, axis=1)  # (D, W)
    zy_proj = np.max(aggregated, axis=2)  # (D, H)


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(xy_proj, cmap='viridis', origin='lower')
    axes[0].set_title("XY Projection")
    
    axes[1].imshow(xz_proj, cmap='viridis', origin='lower')
    axes[1].set_title("XZ Projection")
    
    axes[2].imshow(zy_proj, cmap='viridis', origin='lower')
    axes[2].set_title("ZY Projection")
    
    plt.savefig('voxel_projections.png')
    plt.show()