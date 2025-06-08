from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def validate(model, dataloader, num_classes, criterion, device):
    model.eval()
    total_loss = 0.0
    num_classes = dataloader.dataset
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