# PyTorch Image Classification Implementation (Mac-Friendly) — and CUDA Notes

- **Author:** Shahzad Ali - Leader Solutions Architects
- **Purpose:** A small, ready-to-run implementation you can demo to customers to explain **ML inference** on a Mac (Apple Silicon) using **PyTorch + Metal (MPS)**, plus guidance on **CUDA** and how to test it (via cloud/Linux), and resources to position yourself for NVIDIA-related work.

---

##  What This Repo Demonstrates
- A **minimal PyTorch image-classification POC** using a pre-trained **ResNet‑50**.
- **GPU acceleration on Mac** via **Metal Performance Shaders (MPS)** backend (works on Apple Silicon M‑series).  
- Clear explanation of **inference** vs **training**, and how to run this locally or in the cloud.  
- Why **CUDA** is *not supported* on macOS and how to still test GPU workloads.

> **Note:** macOS no longer supports running or developing CUDA applications. The NVIDIA CUDA Toolkit 11.6 docs state: *“no longer supports development or running applications on macOS.”*   
> For GPU acceleration on Mac, use **PyTorch MPS** or **TensorFlow with `tensorflow-metal`**. 

---

##  Quick Primer: GPU vs CUDA vs Metal (MPS)
- **GPU**: Hardware accelerator (NVIDIA, AMD, Apple) with thousands of cores suited for parallel math.  
- **CUDA**: NVIDIA’s parallel computing platform & programming model for **NVIDIA GPUs** only. (See the official list of CUDA-capable GPUs.)
- **Metal (MPS)**: Apple’s GPU API + **Metal Performance Shaders** for compute & ML on Mac GPUs. PyTorch exposes MPS as a **`mps` device**, enabling tensor ops and models to run on the Mac GPU. 

**Bottom line for Mac users:** You **cannot** run CUDA natively on modern macOS. Use **MPS for PyTorch** or **`tensorflow-metal`** for TensorFlow to get GPU acceleration.

---

## Prerequisites (Mac / Apple Silicon)
- macOS 12.3+ (for MPS), Python 3.8+ (PyTorch 1.12+ / 2.x). 
- Install PyTorch & TorchVision:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

> PyTorch’s **MPS backend** provides a `mps` device for Apple Silicon/AMD Macs; see docs for requirements and examples.

---

## POC Script (CLI or Streamlit)

### Option A — Command-line
Save as `pytorch_image_classify.py` and run `python pytorch_image_classify.py`.  
If you omit `--images`, the script **downloads three demo images** and prints **top‑5 predictions** with probabilities.

```python
import argparse, os, sys, time
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Device selection (CUDA -> MPS -> CPU)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    else:
        return torch.device("cpu"), "CPU"

# ImageNet class labels
def load_imagenet_labels():
    weights = ResNet50_Weights.DEFAULT
    return weights.meta["categories"]

# Helpers

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

# Inference

def classify_images(image_paths: List[Path], device: torch.device, device_name: str, topk: int = 5):
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights=weights).to(device)
    model.eval()

    labels = load_imagenet_labels()

    print(f"\n✅ Using device: {device_name}")
    print(f"📦 Model: ResNet-50 (pretrained on ImageNet-1K)\n")

    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224), device=device)
        _ = model(dummy)

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
            continue

        input_tensor = preprocess(img).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        elapsed = (time.time() - start) * 1000
        top_probs, top_idxs = probs.topk(topk)
        top_probs = top_probs.cpu().numpy()
        top_idxs = top_idxs.cpu().numpy()
        print(f"🖼️ Image: {img_path.name}  |  ⏱️ Inference: {elapsed:.2f} ms")
        for rank, (idx, p) in enumerate(zip(top_idxs, top_probs), start=1):
            print(f"  {rank}. {labels[idx]}  —  {p*100:.2f}%")
        print("")

# Entry point

def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Classification POC (ResNet-50)")
    parser.add_argument("--images", type=str, default=None, help="Folder of images (.jpg/.jpeg/.png). If omitted, demo images are downloaded.")
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
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
```

### Option B — Streamlit UI
Minimal app to upload images and see top‑5 predictions.

```python
# streamlit_app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

st.set_page_config(page_title="PyTorch Image Classifier (MPS/CUDA)", page_icon="🧠", layout="centered")
st.title("🧠 PyTorch Image Classifier — ResNet‑50")
st.caption("Runs on CUDA (NVIDIA), MPS (Apple Silicon), or CPU. Pretrained on ImageNet‑1K.")

# Select device automatically
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "CUDA (NVIDIA GPU)"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "MPS (Apple Silicon GPU)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

# Load model and labels
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to(device)
model.eval()
labels = weights.meta["categories"]
preprocess = weights.transforms()

st.success(f"Using device: {device_name}")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)
    inp = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idxs = probs.topk(5)
    st.subheader("Top‑5 Predictions")
    for rank, (idx, p) in enumerate(zip(top_idxs.cpu().numpy(), top_probs.cpu().numpy()), start=1):
        st.write(f"{rank}. **{labels[idx]}** — {p*100:.2f}%")
```

Run the UI:
```bash
pip install streamlit
streamlit run streamlit_app.py
```

Following is the UI that will be displayed
<img width="757" height="520" alt="image" src="https://github.com/user-attachments/assets/b6ba6a6c-5e47-40a7-8283-c88edba826ba" />







---

## 🧪 What is *Inference* (and why customers care)?
**Inference** is using a trained model to make predictions on new data (no weight updates). It should be **fast** and **reliable**: low latency for good UX, and high accuracy for trust. Your POC prints per‑image inference time (ms) and top‑K probabilities, which is perfect for a quick demo.

---

## 💡 Testing CUDA (Even on a Mac)
- **On macOS:** You cannot run CUDA apps natively on modern macOS. (CUDA 10.2 was the last to support macOS; toolkit 11.6 and newer dropped macOS for running/dev.) 
- **Workable alternatives on a Mac:**
  - Use **PyTorch MPS** (this repo) to understand ML workflows and GPU acceleration on Mac. 
  
  - Use **TensorFlow + `tensorflow-metal`** to accelerate Keras models. 
  - For **CUDA testing**, provision a **Linux** VM/container on a cloud GPU (AWS/GCP/Azure) or services like RunPod/Lambda. Install **NVIDIA drivers + CUDA Toolkit**, then run the same PyTorch code with `device="cuda"`.

### Quick CUDA smoke test (Linux/Cloud)
```bash
# Verify NVIDIA driver and GPU
nvidia-smi
# Verify CUDA toolkit
nvcc --version
# PyTorch device check (Python)
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
```

To confirm your GPU’s **compute capability** (architecture features), see NVIDIA’s official list. 

---

## 🛠️ Tips to Boost Relevance for NVIDIA‑Related Work
1. **Add an NVIDIA‑ready Docker image** with CUDA:
   - Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
   - Include PyTorch (CUDA build), sample scripts, and a `README` showing `nvidia-smi`/`torch.cuda` checks.
2. **Include Nsight profiling walkthroughs** (Systems/Compute) for kernel & timeline profiling — even from macOS as host tools for remote profiling. 
3. **Showcase cloud GPU runs**:
   - Notebook with **A100/L4/RTX 40xx** instances comparing throughput/latency vs. MPS on Mac; document configs and metrics.
4. **Demonstrate model serving**:
   - Package your inference as a **FastAPI**/**TorchServe** service; optionally benchmark with **batching** and **quantization**.
5. **Document best practices**:
   - Mixed precision (FP16/TF32), data pipelines, memory tuning; include references to **compute capability** when explaining hardware behavior. 
6. **Link to official NVIDIA resources**:
   - CUDA GPUs & compute capability, NGC catalogs, and toolkit docs.

---

## 🔗 Useful References
- **CUDA on macOS — toolkit notice**: *CUDA Toolkit 11.6 no longer supports development or running applications on macOS.* 

- **Historical note on last macOS support (CUDA 10.2)**:   
- **PyTorch MPS backend docs**: [MPS backend — PyTorch]() • [Apple Developer: Accelerated PyTorch on Mac]()  
- **TensorFlow Metal plugin**: [Apple Developer: tensorflow‑metal]() • [PyPI: tensorflow‑metal]()  
- **CUDA-capable GPUs & compute capability**: [NVIDIA Developer]()

---

##  Repo Structure (suggested)
```
.
├── pytorch_image_classify.py      # CLI POC (ResNet‑50)
├── streamlit_app.py               # Simple UI for uploads/top‑K
├── docker/                        # (Optional) CUDA-enabled Dockerfiles
│   └── Dockerfile
├── notebooks/                     # (Optional) Cloud GPU benchmarks & comparisons
│   └── mps_vs_cuda_bench.ipynb
└── README.md                      # This file
```

---

##  How to Pitch This POC to Customers
- **Explain inference clearly**: real-time predictions, latency, and accuracy impact on UX.  
- **Show cross‑environment portability**: the same PyTorch code runs on **CPU/MPS/CUDA** by changing the device.  
- **Demonstrate quick wins**: pre-trained models → immediate value; then suggest fine‑tuning for their data.

---

##  Quick Start
```bash
# Clone and run (Mac / Apple Silicon)
pip install -U pip torch torchvision torchaudio streamlit
python pytorch_image_classify.py  # CLI demo
streamlit run streamlit_app.py    # Web UI demo
```

---

## 📄 License
MIT — feel free to adapt for your own demos and customer POCs.
