# -*- coding: utf-8 -*-
"""Attention U-Net training and recursive fracture-mask prediction."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .config import (
    MODEL_PATH,
    SEGMENTATION_ARTIFACT_PATH,
    TRAINING_DATA_PATH,
    ensure_artifact_directories,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOGGER = logging.getLogger(__name__)


class CrackDataset(Dataset):
    """Pair ``images/<stem>.*`` with ``masks/<stem>_mask.png``."""

    def __init__(self, root=TRAINING_DATA_PATH, resize=(256, 256), augment=False):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.resize = tuple(resize)
        self.augment = augment
        if not self.img_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Training data must contain images and masks directories: {self.root}"
            )

        self.samples = []
        for image_path in sorted(self.img_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                mask_path = self.mask_dir / f"{image_path.stem}_mask.png"
                if mask_path.is_file():
                    self.samples.append((image_path, mask_path))
        if not self.samples:
            raise RuntimeError(f"未找到有效的图像/掩码配对：{self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize(self.resize, Image.Resampling.BILINEAR)
        mask = mask.resize(self.resize, Image.Resampling.NEAREST)
        if self.augment and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.augment and random.random() < 0.2:
            image, mask = TF.vflip(image), TF.vflip(mask)
        image = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask = (TF.to_tensor(mask) >= 0.5).float()
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionBlock(512, 512, 256)
        self.conv_up4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionBlock(256, 256, 128)
        self.conv_up3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionBlock(128, 128, 64)
        self.conv_up2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.conv_up1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))
        u4 = self.up4(c5)
        u4 = self.conv_up4(torch.cat([u4, self.att4(u4, c4)], dim=1))
        u3 = self.up3(u4)
        u3 = self.conv_up3(torch.cat([u3, self.att3(u3, c3)], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv_up2(torch.cat([u2, self.att2(u2, c2)], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv_up1(torch.cat([u1, self.att1(u1, c1)], dim=1))
        return torch.sigmoid(self.final(u1))


def segmentation_loss(prediction, target, positive_weight=5.0):
    """BCE preserves pixel calibration; Dice counters crack/background imbalance."""
    probability = prediction.clamp(1e-6, 1.0 - 1e-6)
    bce = -(
        positive_weight * target * torch.log(probability)
        + (1.0 - target) * torch.log(1.0 - probability)
    ).mean()
    intersection = torch.sum(prediction * target, dim=(1, 2, 3))
    denominator = torch.sum(prediction, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * bce + 0.5 * dice


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    train_dir=TRAINING_DATA_PATH,
    model_save_path=MODEL_PATH,
    epochs=100,
    batch_size=4,
    val_ratio=0.2,
    learning_rate=1e-4,
    seed=42,
):
    """Train AU-Net and save the checkpoint with the best validation F1."""
    _seed_everything(seed)
    ensure_artifact_directories()
    train_dataset = CrackDataset(train_dir, augment=True)
    val_dataset = CrackDataset(train_dir, augment=False)
    if len(train_dataset) < 2:
        raise RuntimeError("训练样本不足，至少需要2组图像/掩码")
    val_size = max(1, int(round(len(train_dataset) * val_ratio)))
    train_size = len(train_dataset) - val_size
    indices = torch.randperm(
        len(train_dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = AttentionUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                predictions = model(images)
            loss = segmentation_loss(predictions.float(), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        tp = fp = fn = correct = total = 0
        with torch.inference_mode():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    predictions = model(images)
                batch_val_loss = segmentation_loss(predictions.float(), masks)
                val_loss += batch_val_loss.item() * images.size(0)
                pred = predictions >= 0.5
                truth = masks >= 0.5
                tp += torch.logical_and(pred, truth).sum().item()
                fp += torch.logical_and(pred, ~truth).sum().item()
                fn += torch.logical_and(~pred, truth).sum().item()
                correct += (pred == truth).sum().item()
                total += truth.numel()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_size,
            "val_loss": val_loss / val_size,
            "pixel_accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        history.append(row)
        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]["lr"]
        row["learning_rate"] = current_lr
        LOGGER.info(
            "Epoch %s/%s train=%.4f val=%.4f PA=%.4f F1=%.4f lr=%.2e",
            epoch,
            epochs,
            row["train_loss"],
            row["val_loss"],
            row["pixel_accuracy"],
            f1,
            current_lr,
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_save_path)

    SEGMENTATION_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    (SEGMENTATION_ARTIFACT_PATH / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Best checkpoint saved: %s; validation F1=%.4f", model_save_path, best_f1)
    return Path(model_save_path)


def _load_model(model_path, device):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}. Run `borehole-fracture train`."
        )
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = AttentionUNet().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_images_in_folder(
    src_folder, dest_folder, model_path=MODEL_PATH, resize=(256, 256), threshold=0.5
):
    """Predict images recursively and preserve the source subdirectory layout."""
    src_folder, dest_folder = Path(src_folder), Path(dest_folder)
    if not src_folder.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{src_folder}")
    files = sorted(
        p for p in src_folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"输入目录中没有图像：{src_folder}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)
    saved = []
    for index, image_path in enumerate(files, 1):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        tensor = (
            TF.normalize(
                TF.to_tensor(image.resize(tuple(resize), Image.Resampling.BILINEAR)),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            .unsqueeze(0)
            .to(device)
        )
        with torch.inference_mode():
            prediction = model(tensor).cpu().squeeze().numpy()
        binary = (prediction >= threshold).astype(np.uint8)
        binary_image = Image.fromarray(binary * 255).resize(original_size, Image.Resampling.NEAREST)
        output = (1 - (np.asarray(binary_image) // 255).astype(np.uint8)) * 255
        save_dir = dest_folder / image_path.relative_to(src_folder).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{image_path.stem}_mask.png"
        Image.fromarray(output).save(save_path)
        saved.append(save_path)
        LOGGER.info("[%s/%s] saved=%s", index, len(files), save_path)
    return saved


def build_parser():
    parser = argparse.ArgumentParser(description="Attention U-Net fracture segmentation")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="训练并保存模型")
    train.add_argument("--train-dir", type=Path, default=TRAINING_DATA_PATH)
    train.add_argument("--model", type=Path, default=MODEL_PATH)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4)
    predict = sub.add_parser("predict", help="递归生成二值掩码")
    predict.add_argument("src", type=Path)
    predict.add_argument("dest", type=Path)
    predict.add_argument("--model", type=Path, default=MODEL_PATH)
    predict.add_argument("--threshold", type=float, default=0.5)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train_model(args.train_dir, args.model, args.epochs, args.batch_size)
    else:
        predict_images_in_folder(args.src, args.dest, args.model, threshold=args.threshold)


if __name__ == "__main__":
    main()
