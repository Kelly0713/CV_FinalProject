import argparse
import csv
import json
import os
import sys
from PIL import Image
import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from model import build_model

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(input_dir: str):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                image_paths.append(os.path.join(root, fname))
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    config_path = os.path.join(BASE_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt_path = os.path.join(BASE_DIR, cfg["checkpoint_file"])
    arch = cfg["architecture"]
    class_order = cfg["class_order"]
    input_size = tuple(cfg["input_size"])
    mean = cfg["normalization_mean"]
    std = cfg["normalization_std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(arch=arch, strategy="full_fine_tuning", num_classes=len(class_order), pretrained=False)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image_paths = collect_images(args.input_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "prediction"])

        with torch.no_grad():
            for img_path in image_paths:
                image_id = os.path.splitext(os.path.basename(img_path))[0]
                img = Image.open(img_path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                logits = model(tensor)
                pred_idx = int(torch.argmax(logits, dim=1).item())
                writer.writerow([image_id, class_order[pred_idx]])


if __name__ == "__main__":
    main()