import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import build_model, set_backbone_trainable, count_parameters, CLASS_ORDER


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(input_size=(224, 224)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, val_transform


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to STUDENT_PUBLIC/classification")
    parser.add_argument("--arch", type=str, default="densenet121", choices=["resnet34", "densenet121"])
    parser.add_argument("--strategy", type=str, default="full_fine_tuning", choices=["feature_extraction", "full_fine_tuning"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="experiments")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "validation"), transform=val_tf)

    assert train_dataset.classes == CLASS_ORDER, f"Class mismatch: {train_dataset.classes}"

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(arch=args.arch, strategy=args.strategy, num_classes=len(CLASS_ORDER), pretrained=True)
    model.to(device)

    # 階段一：先凍結 backbone，用 1e-3 訓練分類頭 (Warmup)
    set_backbone_trainable(model, args.arch, trainable=False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    total_p, train_p = count_parameters(model)
    print(f"[{args.arch} | {args.strategy}] Total Params: {total_p:,} | Initial Trainable: {train_p:,}")

    exp_name = f"{args.arch}_{args.strategy}"
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 階段二：若是 full_fine_tuning 且過了 warmup_epochs，解凍全網路並降低 LR
        if args.strategy == "full_fine_tuning" and epoch == args.warmup_epochs + 1:
            set_backbone_trainable(model, args.arch, trainable=True)
            total_p, train_p = count_parameters(model)
            print(f"--- Epoch {epoch}: Unfreezing full backbone! Trainable Params: {train_p:,} | LR: {args.lr} ---")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        _, _, train_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average="macro", zero_division=0
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        dt = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)

    file_size_mb = os.path.getsize(best_ckpt_path) / (1024 * 1024)
    print(f"\nFinished {exp_name} | Best Val Macro F1: {best_val_f1:.4f} | Model Size: {file_size_mb:.2f} MB")

    with open(os.path.join(args.save_dir, f"{exp_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()