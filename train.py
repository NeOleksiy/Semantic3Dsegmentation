import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils.logging import setup_logger
from evaluation.metrics import AverageMeter,calculate_iou
from evaluation.visualize import plot_segmentation_metrics
from models import model_factory
from data.dataloader import get_dataloader
from sklearn.metrics import confusion_matrix
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))


@hydra.main(config_path="./configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()
    logger.info(f"Calculating device: {device}")

    OmegaConf.resolve(cfg)

    train_loader = get_dataloader(cfg, split='train')
    val_loader = get_dataloader(cfg, split='val')

    model = model_factory(cfg).to(device)
    
    logger.info(f"Init {cfg['model']['name']} model")
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs
    )
    
    criterion = nn.CrossEntropyLoss()
    
    if os.path.isdir(cfg.output.log_dir):
        shutil.rmtree(cfg.output.log_dir)
        os.makedirs(cfg.output.log_dir)
    
    if not os.path.isdir(cfg.output.checkpoint_dir):
        os.makedirs(cfg.output.checkpoint_dir)

    if not os.path.isdir(cfg.output.saved_model):
        os.makedirs(cfg.output.saved_model)
    
    writer = SummaryWriter(cfg.output.log_dir)

    train_loss = []
    test_loss_m = []
    test_miou_list = []
    train_loss_meter = AverageMeter()
    test_loss_meter = AverageMeter()

    best_mIoU = -1
    no_improvement_counter = 0
    patience = 10

    epoch = 1 

    logger.info("Start Train")
    for epoch in range(epoch, cfg.training.epochs):
        model.train()
        train_loss_meter.reset()

        progress_bar = tqdm(train_loader, colour='cyan')

        for i, (feature, mask) in enumerate(progress_bar):

            feature, mask = feature.to(device), mask.to(device)

            output = model(feature)
            
            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())
            progress_bar.set_description(f"Train | Epoch {epoch}/{cfg.training.epochs} | Loss {train_loss_meter.avg:.4f} | lr {optimizer.param_groups[0]['lr']}")
            writer.add_scalar("Train/Loss AVG", train_loss_meter.avg, epoch * len(train_loader) + i)
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)

        test_loss, test_acc, test_miou = validate(model, val_loader, cfg.model.num_classes, criterion, device)
        logger.info(f"Val | Loss {test_loss:.4f} | mIoU: {test_miou:.4f}% | Accuracy: {test_acc:.3f}")
        writer.add_scalar("Val/Loss", test_loss, epoch)
        writer.add_scalar("Val/mIoU", test_miou, epoch)
        writer.add_scalar("Val/Accuracy", test_acc, epoch)

        scheduler.step()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "mIoU": test_miou
        }

        torch.save(checkpoint, os.path.join(cfg.output.saved_model, "last.pt"))
        
        train_loss.append(train_loss_meter.avg)
        test_miou_list.append(test_miou)
        test_loss_m.append(test_loss)

        if best_mIoU < test_miou:
            best_mIoU = test_miou
            torch.save(checkpoint, os.path.join(cfg.output.saved_model, "best.pt"))
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
        
        # # Early Stopping
        # if no_improvement_counter > patience:
        #     print("Early stopping!")
        #     break
    
    plot_segmentation_metrics(train_loss=train_loss, val_loss=test_loss_m,val_mean_iou=test_miou_list)

def validate(model, dataloader, num_classes, criterion, device):
    model.eval()
    total_loss = 0.0
    num_classes = len(dataloader.dataset.allowed_labels)
    cm = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for points, labels in dataloader:
            feature, mask = points.to(device), labels.to(device)

            # feature = PointRCNNInput([feature])
            output = model(feature)
            loss = criterion(output, mask)
            
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            cm += confusion_matrix(mask.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), labels=range(num_classes))

    return total_loss/len(dataloader), cm.diagonal().sum()/cm.sum(), calculate_iou(cm)

if __name__ == "__main__":
    main()