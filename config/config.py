TRAIN_IMAGE_DIR = "/Users/choetaewon/Documents/GitHub/UNET/datasets/train_DS/train"
TRAIN_MASK_DIR = "/Users/choetaewon/Documents/GitHub/UNET/datasets/train_DS/train_labels"

TEST_IMAGE_DIR = "/Users/choetaewon/Documents/GitHub/UNET/datasets/test_DS/test"
TEST_MASK_DIR = "/Users/choetaewon/Documents/GitHub/UNET/datasets/test_DS/test_labels"

VAL_IMAGE_DIR = "/Users/choetaewon/Documents/GitHub/UNET/datasets/val_DS/val"
VAL_MASK_DIR = "/Users/choetaewon/Documents/GitHub/UNET/datasets/val_DS/val_labels"

EPOCH = 20
LR = 1e-4
BATCH_SIZE = 16
DEVICE = "mps"

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(model, val_loader, device):
    model.eval()  # 1. 평가 모드 ON (Dropout, BatchNorm 등을 평가용으로 고정)

    with torch.no_grad():  # 2. 미분 계산 OFF (메모리 폭발 방지, 속도 향상)
        # 배치 하나 꺼내오기
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)

        # 3. 모델에 넣고 날것의 예측값(Logits) 뽑기
        outputs = model(images)

        # 4. 핵심! Logits -> 확률(Sigmoid) -> 이진 마스크(0 or 1) 변환
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()  # 0.5 넘으면 1.0(건물), 아니면 0.0(배경)

        # 5. 화면에 그리기 위해 CPU로 내리고 Numpy로 변환 (첫 번째 사진만)
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        mask_np = masks[0].permute(1, 2, 0).cpu().numpy()
        pred_np = preds[0].permute(1, 2, 0).cpu().numpy()

        # 원본 이미지 정규화 복원 (Denormalize)
        img_np = (img_np * 0.5) + 0.5
        img_np = np.clip(img_np, 0, 1)

        # 6. Matplotlib으로 3장 나란히 예쁘게 띄우기
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("1. Input Image")
        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("2. Ground Truth (Real Mask)")
        plt.imshow(mask_np.squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("3. Model Prediction")
        plt.imshow(pred_np.squeeze(), cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    model.train()  # 시각화 끝났으면 다시 학습 모드로 돌려놓는 센스!

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1. 모델이 뱉은 날것(Logits)을 0~1 사이 확률로 압축
        inputs = torch.sigmoid(inputs)

        # 2. 복잡한 차원(Batch, C, H, W) 다 무시하고 1차원으로 쫙 펴기
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 3. 모델의 예측과 정답지의 교집합 계산
        intersection = (inputs * targets).sum()

        # 4. Dice 공식 (분모 0 되는 거 막기 위해 smooth 추가)
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # 5. 오차(Loss)는 작아져야 하므로 1에서 빼기
        return 1 - dice