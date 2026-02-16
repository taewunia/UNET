import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

EPOCH = 50
LR = 1e-3
BATCH_SIZE = 64

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            convlayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                bias=bias),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU()
                                      )
            return convlayer

        self.encoder1_1 = CBR2d(in_channels=3, out_channels=6)
        self.encoder1_2 = CBR2d(in_channels=6, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2_1 = CBR2d(in_channels=64, out_channels=128)
        self.encoder2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3_1 = CBR2d(in_channels=128, out_channels=256)
        self.encoder3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder4_1 = CBR2d(in_channels=256, out_channels=512)
        self.encoder4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.encoder5_1 = CBR2d(in_channels=512, out_channels=1024)


        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512,
                                          out_channels=512,
                                          kernel_size=2,
                                          stride=2,
                                          padding=0)

        self.dec4_2 = CBR2d(in_channels=1024, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256,
                                          out_channels=256,
                                          kernel_size=2,
                                          stride=2,
                                          padding=0)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels= 256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128,
                                          out_channels=128,
                                          kernel_size=2,
                                          stride=2,
                                          padding=0)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=64,
                                          kernel_size=2,
                                          stride=2,
                                          padding=0)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=32)

        self.fc = nn.Conv2d(in_channels=32,
                            out_channels=2,
                            kernel_size=1,
                            stride=1,
                            padding=0)

    def forward(self, x):
        encode1_1 = self.encoder1_1(x)
        encode1_2 = self.encoder1_2(encode1_1)
        pool1 = self.pool1(encode1_2)

        encode2_1 = self.encoder2_1(pool1)
        encode2_2 = self.encoder2_2(encode2_1)
        pool2 = self.pool2(encode2_2)

        encode3_1 = self.encoder3_1(pool2)
        encode3_2 = self.encoder3_2(encode3_1)
        pool3 = self.pool3(encode3_2)

        encode4_1 = self.encoder4_1(pool3)
        encode4_2 = self.encoder4_2(encode4_1)
        pool4 = self.pool4(encode4_2)

        encode5_1 = self.encoder5_1(pool4)

        decode4_1 = self.dec4_1(encode5_1)
        decode4_2 = self.dec4_2(decode4_1)
        unpool4 = self.unpool4(decode4_2)

        cat3 = torch.cat((unpool4, encode4_2), dim=1)

        decode3_2 = self.dec3_2(cat3)
        decode3_1 = self.dec3_1(decode3_2)

        unpool3 = self.unpool3(decode3_1)

        cat2 = torch.cat((unpool3, encode3_2), dim=1)

        decode2_2 = self.dec2_2(cat2)
        decode2_1 = self.dec2_1(decode2_2)

        unpool2 = self.unpool2(decode2_1)

        cat1 = torch.cat((unpool2, encode2_2), dim=1)

        decode1_2 = self.dec1_2(cat3)
        decode1_1 = self.dec1_1(decode1_2)

        x = self.fc(decode1_1)

        return x

model = UNet().to(device)
print(model)
