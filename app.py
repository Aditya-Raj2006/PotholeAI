
import os
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from PIL import Image
from torchvision import transforms

st.set_page_config(
    page_title="PotholeAI | Research Platform",
    page_icon="🛣️",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f5f8fc;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    background: linear-gradient(120deg, #102b46, #176b87);
    padding: 30px; border-radius: 18px; color: white;
    margin-bottom: 22px;
}
.hero h1 {color: white; margin-bottom: 6px;}
.hero p {color: #e4f2fa; font-size: 17px;}
.result-card {
    background: white; padding: 18px; border-radius: 12px;
    border: 1px solid #dce6ef; margin-bottom: 12px;
}
.small-note {color: #526579; font-size: 13px;}
</style>
<div class="hero">
<h1>🛣️ PotholeAI</h1>
<p>AI-Powered Pothole Segmentation & Research Assessment</p>
<p>Computer vision for road-surface inspection and pothole analysis</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "pothole_pretrained_unet.pth"
)
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model checkpoint is missing.")
    net = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    weights = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    net.load_state_dict(weights)
    net.to(DEVICE)
    net.eval()
    return net

@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Image Analysis", "About the Project", "Limitations"]
)

if page == "Image Analysis":
    st.subheader("Upload and analyze a road image")
    st.write(
        "Upload a JPG or PNG photograph. The trained model will "
        "predict a pothole-like region and generate a visual overlay."
    )
    uploaded = st.file_uploader(
        "Choose a road image", type=["jpg", "jpeg", "png"]
    )
    threshold = st.slider(
        "Segmentation threshold", 0.10, 0.90, 0.50, 0.05
    )

    if uploaded is not None:
        original = Image.open(uploaded).convert("RGB")
        st.image(original, caption="Uploaded road image",
                 use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            try:
                with st.spinner("Analyzing image..."):
                    model = load_model()
                    x = get_transform()(original).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits = model(x)
                        probability = torch.sigmoid(logits)
                        probability = F.interpolate(
                            probability,
                            size=(original.height, original.width),
                            mode="bilinear",
                            align_corners=False
                        )[0, 0].cpu().numpy()

                    mask = (probability >= threshold)
                    mask_img = Image.fromarray(
                        (mask * 255).astype(np.uint8)
                    )

                    arr = np.asarray(original).copy()
                    overlay = arr.copy()
                    overlay[mask] = (
                        0.55 * arr[mask]
                        + 0.45 * np.array([255, 45, 45])
                    ).astype(np.uint8)

                    fraction = 100 * float(mask.mean())
                    status = (
                        "Pothole-like region segmented"
                        if mask.any()
                        else "No region detected at this threshold"
                    )

                st.success("Analysis completed")
                c1, c2, c3 = st.columns(3)
                c1.metric("Detection result",
                          "Region found" if mask.any() else "Not detected")
                c2.metric("Segmented pixels", f"{int(mask.sum()):,}")
                c3.metric("Image-area fraction", f"{fraction:.2f}%")

                left, middle, right = st.columns(3)
                left.image(original, caption="Original",
                           use_container_width=True)
                middle.image(mask_img, caption="Predicted mask",
                             use_container_width=True)
                right.image(Image.fromarray(overlay),
                            caption="Overlay", use_container_width=True)

                st.subheader("Analysis report")
                st.write(status)
                st.write(f"Threshold used: {threshold:.2f}")
                st.write(f"Model input resolution: {SIZE} × {SIZE}")
                st.download_button(
                    "Download segmentation mask",
                    data=mask_img.convert("L").tobytes(),
                    file_name="pothole_mask_raw_pixels.bin",
                    mime="application/octet-stream"
                )
                st.caption(
                    "The downloadable file contains raw mask pixels, "
                    "not a standard image file."
                )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

elif page == "About the Project":
    st.subheader("About PotholeAI")
    st.write("""
    **Research topic:** A Reflection-Resilient and Metric-Depth-Aware
    Deep Learning Framework for Pothole Severity Assessment.

    This prototype uses a U-Net architecture with a ResNet-34 encoder
    and a model trained using the Pothole440 dataset.

    Current application: image-based pothole-region segmentation,
    mask visualization, and image-area fraction reporting.
    """)

elif page == "Limitations":
    st.subheader("Research limitations")
    st.warning("""
    Image-area fraction is not physical road-surface area.
    The current prototype does not yet provide calibrated physical
    depth or volume, validated engineering severity grades,
    reflection-specific robustness validation, or video tracking.
    These features require additional data and testing.
    """)
    st.write(
        "Predictions should be reviewed by a human and should not "
        "be treated as certified road-safety measurements."
    )

st.divider()
st.caption(
    "PotholeAI Research Prototype | Computer Vision for Road Inspection"
)
