import sys
import torch
sys.path.append("pvcnn")
from models.external.pvcnn.models.s3dis import PVCNN2

device = "cuda" if torch.cuda.is_available() else "cpu"
PVCNN = PVCNN2