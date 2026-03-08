import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, mask):
        pred = pred.view(-1)
        mask = mask.view(-1)
        pred = torch.sigmoid(pred)

        dice  = (2 * (pred * mask).sum() + self.smooth) / (pred.sum() + mask.sum()+ self.smooth)

        return 1 - dice


class MultiDiceLoss(nn.Module):
    def __init(self, smooth=1e-6):
        super().__init__()


def visualize_prediction(model, device, data):
    model.eval()
    with torch.no_grad():
        model = model.to(device)
        input, mask = next(iter(data))
        input = input[0]
        mask = mask[0]
        input = input.to(device)
        input = input.unsqueeze(dim=0)
        pred = model(input)    # c h w
        pred = pred.sigmoid()
        pred = (pred > 0.5).float()

        input = input.squeeze(0)
        pred = pred.squeeze(0)
        pred = np.array(pred.permute(1, 2, 0).cpu()) # h w c
        input = np.array(input.permute(1, 2, 0).cpu())
        mask = np.array(mask.permute(1, 2, 0).cpu())

        pred = (pred * 0.5) + 0.5
        input = (input * 0.5) + 0.5
        mask = (mask * 0.5) + 0.5

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("1. Input Image")
        plt.imshow(input)
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







