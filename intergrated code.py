import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import numpy as np

EPOCH = 50
LR = 1e-3
BATCH_SIZE = 16

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

train_image_dir = "/Users/choetaewon/Documents/GitHub/UNET/datasets/train_DS/train"
train_mask_dir="/Users/choetaewon/Documents/GitHub/UNET/datasets/train_DS/train_labels"
val_image_dir = "/Users/choetaewon/Documents/GitHub/UNET/datasets/val_DS/val"
val_mask_dir = "/Users/choetaewon/Documents/GitHub/UNET/datasets/val_DS/val_labels"

train_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

class Dataset(Dataset):
    def __init__(self, image_dir=None, mask_dir=None, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0

        if self.transform is not None:
            algumentation = self.transform(image=image, mask=mask)
            image = algumentation["image"]
            mask = algumentation["mask"]
            mask = mask.clone().detach().float()
            mask = torch.unsqueeze(mask, dim=0)

        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, smooth):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        pred = pred.view(-1)
        mask = mask.view(-1)
        pred = pred.sigmoid()
        dice = 2.0 * ((pred * mask).sum() + self.smooth) / ((pred + mask).sum() + self.smooth)

        return 1.0 - dice

def visualize_prediction(model=None, device=None, DL=None):
    image, mask = next(iter(DL))
    image = image[0]
    mask = mask[0]
    image = image.unsqueeze(dim=0) # b, c, h, w
    image = image.to(device)
    model.eval()
    model = model.to(device)
    pred = model(image)
    pred = pred.sigmoid()
    pred = (pred>0.5).float()    # b, c, h, w

    image = image.squeeze(dim=0)# c, h, w
    pred = pred.squeeze(dim=0) # c, h, w

    image = image.detach().cpu().permute(1, 2, 0).numpy()
    mask = mask.detach().cpu().permute(1, 2, 0).numpy()
    pred = pred.detach().cpu().permute(1, 2, 0).numpy()

    image = (image * 0.5) + 0.5
    mask = (mask * 0.5) + 0.5
    pred = (pred * 0.5) + 0.5

    image = np.clip(image, 0, 1)
    mask = np.clip(mask, 0, 1)
    pred = np.clip(pred, 0, 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("1. Input Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Ground Truth (Real Mask)")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Model Prediction")
    plt.imshow(pred.squeeze(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    model.train()

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # ---------------- Encoder ----------------
        self.enc1_1 = CBR2d(3, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        # ---------------- Bottleneck ----------------
        self.enc5_1 = CBR2d(512, 1024)
        self.enc5_2 = CBR2d(1024, 1024)

        # ---------------- Decoder ----------------
        self.unpool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4_2 = CBR2d(1024, 512) # Cat 이후: 512(밑에서) + 512(옆에서) = 1024
        self.dec4_1 = CBR2d(512, 512)

        self.unpool3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_2 = CBR2d(512, 256)
        self.dec3_1 = CBR2d(256, 256)

        self.unpool2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_2 = CBR2d(256, 128)
        self.dec2_1 = CBR2d(128, 128)

        self.unpool1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_2 = CBR2d(128, 64)
        self.dec1_1 = CBR2d(64, 64)


        self.fc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1_2(self.enc1_1(x))
        enc2 = self.enc2_2(self.enc2_1(self.pool1(enc1)))
        enc3 = self.enc3_2(self.enc3_1(self.pool2(enc2)))
        enc4 = self.enc4_2(self.enc4_1(self.pool3(enc3)))

        bottleneck = self.enc5_2(self.enc5_1(self.pool4(enc4)))

        unpool4 = self.unpool4(bottleneck)
        cat4 = torch.cat((unpool4, enc4), dim=1) # 차원 결합
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat((unpool3, enc3), dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat((unpool2, enc2), dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat((unpool1, enc1), dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        output = self.fc(dec1)
        return output

model = UNet().to(device)
dummy_input = torch.randn(1, 3, 256, 256).to(device)
dummy_output = model(dummy_input)

print(f"입력 차원: {dummy_input.shape}")
print(f"출력 차원: {dummy_output.shape} -> (Batch, Channel, Height, Width)")
print("텐서 충돌 X")

train_DS = Dataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=train_transform)
val_DS = Dataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=val_transform)

train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_DL = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

optimizer = optim.Adam(model.parameters(), lr=LR)

criterion_BCE = nn.BCEWithLogitsLoss()
criterion_DICE = DiceLoss(smooth=1e-6)

train_history = []
val_history = []

model.train()
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
        train_loss_bce = criterion_BCE(train_pred, train_labels)
        train_loss_dice = criterion_DICE(train_pred, train_labels)
        loss = train_loss_bce + train_loss_dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        tqdm.set_postfix(train_bar, loss=loss.item())
    avg_train_loss = total_train_loss / len(train_bar)
    train_history.append(avg_train_loss)
    model.eval()
    with torch.no_grad():
        for val_batch, val_labels in val_bar:
            val_batch, val_labels = val_batch.to(device), val_labels.to(device)
            val_pred = model(val_batch)
            val_loss_bce = criterion_BCE(val_pred, val_labels)
            val_loss_dice = criterion_DICE(val_pred, val_labels)
            loss = val_loss_bce + val_loss_dice
            total_val_loss += loss.item()
            tqdm.set_postfix(val_bar, loss=loss.item())
    avg_val_loss = total_val_loss / len(val_bar)
    val_history.append(avg_val_loss)
    print(f'.\navg_train_loss: {avg_train_loss:.4f}, avg_val_loss: {avg_val_loss:.4f}')
    print('-'*40)
    visualize_prediction(model, device, val_DL)

plt.plot(range(1, len(train_history) + 1), train_history, label='train loss', color='green')
plt.title("avg_train_loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.legend()
plt.show()

plt.plot(range(1, len(val_history) + 1), val_history, label='val loss', color='red')
plt.title("avg_val_loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.legend()
plt.show()

