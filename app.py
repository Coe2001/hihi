import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import os

# --- SETTINGS ---
MODEL_PATH = "mobilenetv2_banknote_predictor.h5"
NUM_CLASSES = 7
CLASS_NAMES = ["50", "100", "200", "500", "1000", "5000", "10000"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = EfficientNet.from_name("efficientnet-b0")
    model._fc = torch.nn.Linear(model._fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- Streamlit App ---
st.set_page_config(page_title="Banknote Classifier", layout="centered")
st.title("💵 Myanmar Banknote Recognition")
st.caption("Upload an image of a Myanmar banknote to identify its denomination.")

uploaded_file = st.file_uploader("📤 Upload a banknote image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Banknote", use_column_width=True)

    with st.spinner("🔍 Predicting..."):
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        pred_label = CLASS_NAMES[pred_class]
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    st.success(f"✅ Predicted Denomination: **{pred_label} Ks**")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")
