# streamlit_app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

st.set_page_config(page_title="PyTorch Image Classifier (MPS/CUDA)", page_icon="🧠", layout="centered")
st.title("🧠 PyTorch Image Classifier — ResNet‑50")
st.caption("Runs on CUDA (NVIDIA), MPS (Apple Silicon), or CPU. Pretrained on ImageNet‑1K.")

# Auto-select device
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
