import argparse
import json
import os
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

from model import build_model, count_parameters, CLASS_ORDER


def plot_curves(history_path: str, save_path: str, title: str):
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_f1"], label="Train Macro F1")
    ax2.plot(epochs, history["val_f1"], label="Val Macro F1")
    ax2.set_title(f"{title} - Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True, choices=["resnet34", "densenet121"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--history", type=str, default=None, help="Path to *_history.json")
    parser.add_argument("--save_cm", type=str, default="confusion_matrix.png")
    parser.add_argument("--save_curve", type=str, default="learning_curve.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "validation"), transform=val_tf)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(arch=args.arch, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    elapsed_ms = ((time.time() - start_time) / len(val_dataset)) * 1000.0

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    total_p, _ = count_parameters(model)
    size_mb = os.path.getsize(args.ckpt) / (1024 * 1024)

    print(f"=== Evaluation Report: {args.arch} ({args.ckpt}) ===")
    print(f"Total Params : {total_p:,}")
    print(f"File Size    : {size_mb:.2f} MB")
    print(f"Inference    : {elapsed_ms:.2f} ms / image")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro Prec   : {prec:.4f}")
    print(f"Macro Recall : {rec:.4f}")
    print(f"Macro F1     : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30)
    plt.title(f"Confusion Matrix - {args.arch}")
    plt.tight_layout()
    plt.savefig(args.save_cm, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {args.save_cm}")

    if args.history and os.path.exists(args.history):
        plot_curves(args.history, args.save_curve, title=args.arch)


if __name__ == "__main__":
    main()