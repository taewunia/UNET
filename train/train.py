from model.unet import UNet
from config.config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR, TEST_IMAGE_DIR, TEST_MASK_DIR, LR, EPOCH, BATCH_SIZE, DEVICE
from utils.pre_process import PreProcess, train_transform, val_transform
from utils.diceloss_visualize import DiceLoss, visualize_prediction
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

if DEVICE == 'mps':
    device = torch.device('mps')

else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
model = UNet().to(device)

train_DS = PreProcess(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
val_DS = PreProcess(image_dir=VAL_IMAGE_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
test_DS = PreProcess(image_dir=TEST_IMAGE_DIR, mask_dir=TEST_MASK_DIR, transform=val_transform)

train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_DL = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_DL = DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

loss_history = []
val_loss_history = []
for epoch in range(EPOCH):
    model.train()
    train_bar = tqdm(train_DL, desc=f'{epoch+1}/EPOCH', colour='green')
    val_bar = tqdm(val_DL, desc=f'{epoch+1}/EPOCH', colour='red')
    total_train_loss = 0
    avg_train_loss = 0
    total_val_loss = 0
    avg_val_loss = 0
    for train_batch, train_labels in train_bar:
        train_batch, train_labels = train_batch.to(device), train_labels.to(device)
        train_pred = model(train_batch)
        train_loss_bce = criterion_bce(train_pred, train_labels)
        train_loss_dice = criterion_dice(train_pred, train_labels)
        loss = train_loss_bce + train_loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        tqdm.set_postfix(train_bar, loss=loss.item())
    avg_train_loss = total_train_loss / len(train_bar)
    loss_history.append(avg_train_loss)
    model.eval()
    for val_batch, val_labels in val_bar:
        val_batch, val_labels = val_batch.to(device), val_labels.to(device)
        val_pred = model(val_batch)
        val_loss_bce = criterion_bce(val_pred, val_labels)
        val_loss_dice = criterion_dice(val_pred, val_labels)
        loss = val_loss_bce + val_loss_dice
        total_val_loss += loss.item()
        tqdm.set_postfix(val_bar, loss=loss.item())
    avg_val_loss = total_val_loss / len(val_bar)
    val_loss_history.append(avg_val_loss)
    print(f'.\navg_train_loss: {avg_train_loss:.4f}, avg_val_loss: {avg_val_loss:.4f}')
    print('-'*40)
    visualize_prediction(model, device, test_DL)

plt.plot(range(1, len(loss_history) + 1), loss_history, label='train loss', color='green')
plt.title("avg_train_loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.legend()
plt.show()

plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='val loss', color='red')
plt.title("avg_train_loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.legend()
plt.show()
