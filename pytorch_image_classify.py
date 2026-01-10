
# pytorch_image_classify.py
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# ---------------------------
# Device selection (CUDA -> MPS -> CPU)
# ---------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    elif torch.backends.mps.is_available():
        # MPS = Metal Performance Shaders (Apple Silicon GPUs)
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    else:
        return torch.device("cpu"), "CPU"

# ---------------------------
# ImageNet class labels
# ---------------------------
def load_imagenet_labels():
    """
    Returns a list of 1000 ImageNet class labels.
    Uses torchvision's built-in metadata from the weights enum.
    """
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    return categories

# ---------------------------
# Image loading helpers
# ---------------------------
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}

def collect_images_from_folder(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")
    images = [p for p in folder.iterdir() if is_image_file(p)]
    if not images:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png images found in: {folder}")
    return images

def download_demo_images(tmp_dir: Path) -> List[Path]:
    """
    Downloads 3 demo images (dog, cat, car) if --images is not provided.
    """
    import urllib.request

    tmp_dir.mkdir(parents=True, exist_ok=True)
    demos = [
        ("dog.jpg", "https://images.unsplash.com/photo-1507149833265-60c372daea22?w=800"),
        ("cat.jpg", "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=800"),
        ("car.jpg", "https://images.unsplash.com/photo-1493238792000-8113da705763?w=800"),
    ]
    out_paths = []
    for fname, url in demos:
        out_path = tmp_dir / fname
        if not out_path.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, out_path)
        out_paths.append(out_path)
    return out_paths

# ---------------------------
# Inference
# ---------------------------
def classify_images(image_paths: List[Path], device: torch.device, device_name: str, topk: int = 5):
    # Load pretrained ResNet-50 and associated preprocessing
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()  # Proper transforms for ResNet-50 ImageNet
    model = resnet50(weights=weights).to(device)
    model.eval()

    labels = load_imagenet_labels()

    print(f"\n✅ Using device: {device_name}")
    print(f"📦 Model: ResNet-50 (pretrained on ImageNet-1K)\n")

    # Warm-up (helps stabilize GPU/MPS performance)
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224), device=device)
        _ = model(dummy)

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
            continue

        # Preprocess and forward pass
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        start = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        elapsed = (time.time() - start) * 1000  # ms

        # Top-k predictions
        top_probs, top_idxs = probs.topk(topk)
        top_probs = top_probs.cpu().numpy()
        top_idxs = top_idxs.cpu().numpy()

        print(f"🖼️ Image: {img_path.name}  |  ⏱️ Inference: {elapsed:.2f} ms")
        for rank, (idx, p) in enumerate(zip(top_idxs, top_probs), start=1):
            print(f"  {rank}. {labels[idx]}  —  {p*100:.2f}%")
        print("")

# ---------------------------
# Entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Classification POC (ResNet-50)")
    parser.add_argument("--images", type=str, default=None,
                        help="Path to a folder with image files (.jpg/.jpeg/.png). If omitted, downloads demo images.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K predictions to show (default: 5)")
    args = parser.parse_args()

    device, device_name = get_device()

    if args.images:
        folder = Path(args.images)
        image_paths = collect_images_from_folder(folder)
    else:
        print("No --images provided; downloading demo images...")
        tmp_dir = Path("./_demo_images")
        image_paths = download_demo_images(tmp_dir)

    classify_images(image_paths, device, device_name, topk=args.topk)

if __name__ == "__main__":
    # Make sure script prints stack traces on failure
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

