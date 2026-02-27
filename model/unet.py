import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import os
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
print(device)

EPOCH = 50
LR = 1e-3
BATCH_SIZE = 64

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

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = UNet().to(device)
dummy_input = torch.randn(1, 3, 256, 256).to(device)
dummy_output = model(dummy_input)

print(f"입력 차원: {dummy_input.shape}")
print(f"출력 차원: {dummy_output.shape} -> (Batch, Channel, Height, Width)")
print("텐서 충돌 없이 훈련 준비 완벽하게 완료")