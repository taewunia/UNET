from model.unet import UNet
from config.config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR, TEST_IMAGE_DIR, TEST_MASK_DIR, LR, EPOCH, BATCH_SIZE, DEVICE
from utils.pre_process import PreProcess, train_transform, val_transform
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import dataloader, DataLoader
from tqdm import tqdm

if DEVICE == 'mps':
    device = torch.device('mps')

else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
model = UNet().to(device)

train_DS = PreProcess(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
test_DS = PreProcess(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR, transform=val_transform)