"""
Streamlit web app: upload two images, see them side by side, get the
embedding distance and a same/different verdict from the trained
Siamese network.

Usage:
    streamlit run app.py

This opens a local web page (usually http://localhost:8501) in your browser.
"""

import streamlit as st
import torch
from PIL import Image

from model import SiameseNetwork
from data import TRANSFORM


st.set_page_config(page_title="Siamese Character Verifier", page_icon="🔤", layout="centered")


@st.cache_resource
def load_model(model_path: str, device: str):
    model = SiameseNetwork(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess(pil_image: Image.Image):
    gray = pil_image.convert("L")
    tensor = TRANSFORM(gray).unsqueeze(0)  # (1, 1, 105, 105)
    return tensor


def get_distance(model, tensor_a, tensor_b, device):
    with torch.no_grad():
        emb_a = model.embed(tensor_a.to(device))
        emb_b = model.embed(tensor_b.to(device))
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b, p=2).item()
    return distance


# ---- UI ----

st.title("🔤 Siamese Network Character Verifier")
st.caption("Upload two character images. The model predicts whether they're the same character "
           "— even for characters it never saw during training.")

device = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Distance threshold",
        min_value=0.0, max_value=3.0, value=0.6, step=0.01,
        help="Distance below this = predicted SAME. Run find_threshold.py to calibrate "
             "this properly instead of guessing."
    )
    model_path = st.text_input("Model file", value="siamese_model.pt")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Image A", type=["png", "jpg", "jpeg"], key="a")
    if file_a:
        img_a = Image.open(file_a)
        st.image(img_a, caption="Image A", use_container_width=True)

with col2:
    file_b = st.file_uploader("Image B", type=["png", "jpg", "jpeg"], key="b")
    if file_b:
        img_b = Image.open(file_b)
        st.image(img_b, caption="Image B", use_container_width=True)

st.divider()

if file_a and file_b:
    if st.button("Compare", type="primary", use_container_width=True):
        try:
            model = load_model(model_path, device)
        except FileNotFoundError:
            st.error(f"Couldn't find '{model_path}'. Make sure you've run train.py first "
                      "and the .pt file is in this folder.")
            st.stop()

        tensor_a = preprocess(img_a)
        tensor_b = preprocess(img_b)
        distance = get_distance(model, tensor_a, tensor_b, device)

        is_same = distance < threshold

        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Embedding distance", f"{distance:.4f}")
        m2.metric("Threshold", f"{threshold:.2f}")

        if is_same:
            st.success("✅ Predicted: **SAME character**")
        else:
            st.warning("❌ Predicted: **DIFFERENT characters**")

        # simple visual bar showing distance relative to threshold
        st.write("Distance vs. threshold:")
        bar_max = max(distance, threshold) * 1.3
        st.progress(min(distance / bar_max, 1.0))
else:
    st.info("Upload both images to enable comparison.")
