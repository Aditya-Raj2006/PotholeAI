
import os
import io
from collections import deque

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

st.set_page_config(
    page_title="PotholeAI | Research Platform",
    page_icon="🛣️",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 2.35rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
}
.subtitle {
    color: #687386;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

MODEL_URL = (
    "https://github.com/Aditya-Raj2006/PotholeAI/releases/download/v1.1.1/pothole_finetuned_clean_split.pth"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pothole_finetuned_clean_split.pth")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    weights = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )

    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]

    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])



def connected_components(binary_mask, min_pixels=16):
    """Find 8-connected regions and ignore regions smaller than 16 pixels."""
    height, width = binary_mask.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []

            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = cy + dy, cx + dx

                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and binary_mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            if len(pixels) >= min_pixels:
                ys = [p[0] for p in pixels]
                xs = [p[1] for p in pixels]

                components.append({
                    "pixels": len(pixels),
                    "box": (
                        min(xs), min(ys),
                        max(xs), max(ys)
                    ),
                })

    return components


def severity_label(area_percent):
    """
    Provisional image-area categories only.
    These are NOT validated engineering severity grades.
    """
    if area_percent < 0.5:
        return "Low", (45, 155, 75)
    elif area_percent < 2.0:
        return "Moderate", (230, 155, 35)
    return "High", (220, 55, 55)


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.markdown('<div class="main-title">🛣️ PotholeAI</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-assisted pothole segmentation and '
    'preliminary image analysis</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["Image Analysis", "About the Project", "Limitations"],
    )
    st.divider()
    st.caption(f"Device: {DEVICE.upper()}")
    st.caption(f"Model input: {SIZE} × {SIZE}")


if page == "Image Analysis":
    st.header("Pothole detection and severity")
    st.write(
        "Upload a road photograph to segment pothole-like regions, "
        "estimate their count, and view preliminary size categories."
    )

    uploaded = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"],
    )

    threshold = st.slider(
        "Detection threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Higher thresholds generally produce smaller predicted regions.",
    )

    if uploaded is not None:
        try:
            original = ImageOps.exif_transpose(
                Image.open(uploaded)
            ).convert("RGB")
        except Exception:
            st.error("This image could not be opened. Try another image.")
            st.stop()

        st.image(original, caption="Uploaded road image",
                 use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            try:
                with st.spinner("Loading model and analyzing image..."):
                    model = load_model()
                    x = get_transform()(original).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits = model(x)
                        probability = torch.sigmoid(logits)
                        probability = F.interpolate(
                            probability,
                            size=(original.height, original.width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0].cpu().numpy()

                    full_mask = probability >= threshold

                    # Component counting is done at model resolution
                    # to keep processing manageable.
                    small_mask = np.array(
                        Image.fromarray(
                            full_mask.astype(np.uint8) * 255
                        ).resize(
                            (SIZE, SIZE),
                            Image.Resampling.NEAREST,
                        )
                    ) > 0

                    components = connected_components(small_mask)

                    # Ignore tiny regions likely to be segmentation noise.
                    # This filter is a prototype setting, not a validated rule.
                    image_area = original.width * original.height
                    annotated = original.copy()
                    draw = ImageDraw.Draw(annotated)
                    font = ImageFont.load_default()

                    records = []

                    for index, component in enumerate(components, start=1):
                        x1, y1, x2, y2 = component["box"]

                        sx = original.width / SIZE
                        sy = original.height / SIZE

                        box = (
                            int(x1 * sx),
                            int(y1 * sy),
                            int((x2 + 1) * sx),
                            int((y2 + 1) * sy),
                        )

                        # Approximate predicted region size in original pixels.
                        region_fraction = (
                            component["pixels"] / (SIZE * SIZE)
                        )
                        area_percent = region_fraction * 100
                        severity, color = severity_label(area_percent)

                        records.append({
                            "id": index,
                            "severity": severity,
                            "area_percent": area_percent,
                            "color": color,
                            "box": box,
                        })

                        draw.rectangle(box, outline=color, width=4)
                        label = f"#{index} {severity}"
                        draw.text(
                            (box[0] + 3, max(0, box[1] - 15)),
                            label,
                            fill=color,
                            font=font,
                            stroke_width=2,
                            stroke_fill=(255, 255, 255),
                        )

                    mask_image = Image.fromarray(
                        full_mask.astype(np.uint8) * 255
                    )
                    heatmap = Image.fromarray(
                        np.clip(probability * 255, 0, 255).astype(np.uint8)
                    )

                    total = len(records)
                    low_count = sum(r["severity"] == "Low" for r in records)
                    moderate_count = sum(
                        r["severity"] == "Moderate" for r in records
                    )
                    high_count = sum(
                        r["severity"] == "High" for r in records
                    )

                    st.success("Image analysis completed.")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Pothole-like regions", total)
                    c2.metric("Low (provisional)", low_count)
                    c3.metric("Moderate (provisional)", moderate_count)
                    c4.metric("High (provisional)", high_count)

                    st.subheader("Visual results")
                    col1, col2, col3 = st.columns(3)

                    col1.image(
                        original, caption="Original image",
                        use_container_width=True,
                    )
                    col2.image(
                        mask_image, caption="Predicted segmentation mask",
                        use_container_width=True,
                    )
                    col3.image(
                        annotated, caption="Numbered regions and provisional labels",
                        use_container_width=True,
                    )

                    with st.expander("View model probability map"):
                        st.image(
                            heatmap,
                            caption="Grayscale model probability (brighter means higher probability)",
                            use_container_width=True,
                        )

                    st.subheader("Individual region report")

                    if not records:
                        st.warning(
                            "No regions passed the current threshold and "
                            "minimum-size filter. This does not prove the "
                            "road has no potholes."
                        )
                    else:
                        for record in records:
                            st.markdown(
                                f"**Pothole-like region {record['id']}** — "
                                f"{record['severity']} provisional category; "
                                f"approximately {record['area_percent']:.2f}% "
                                "of the image area."
                            )

                    st.info(
                        "Severity labels are preliminary image-area categories. "
                        "They are not physical measurements of pothole depth, "
                        "volume, or engineering severity. One pothole can be "
                        "split into multiple regions, and false detections or "
                        "missed potholes are possible."
                    )

                    report_lines = [
                        "PotholeAI Image Analysis Report",
                        f"Image size: {original.width} x {original.height}",
                        f"Threshold: {threshold:.2f}",
                        f"Detected pothole-like regions: {total}",
                        f"Low provisional categories: {low_count}",
                        f"Moderate provisional categories: {moderate_count}",
                        f"High provisional categories: {high_count}",
                        "",
                        "Individual regions:",
                    ]

                    for record in records:
                        report_lines.append(
                            f"Region {record['id']}: {record['severity']} "
                            f"(image-area estimate "
                            f"{record['area_percent']:.2f}%)"
                        )

                    report_lines += [
                        "",
                        "Limitations: Image-area categories only; not calibrated "
                        "depth, volume, or validated engineering severity.",
                    ]
                    report_text = "\n".join(report_lines)

                    st.subheader("Download results")
                    d1, d2, d3 = st.columns(3)

                    d1.download_button(
                        "Download mask (PNG)",
                        data=png_bytes(mask_image),
                        file_name="pothole_mask.png",
                        mime="image/png",
                    )
                    d2.download_button(
                        "Download annotated image",
                        data=png_bytes(annotated),
                        file_name="pothole_annotated.png",
                        mime="image/png",
                    )
                    d3.download_button(
                        "Download report (TXT)",
                        data=report_text,
                        file_name="pothole_report.txt",
                        mime="text/plain",
                    )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.caption(
                    "If this continues, share a screenshot of the error."
                )


elif page == "About the Project":
    st.header("About the project")
    st.write("""
    **Research topic:** A Reflection-Resilient and Metric-Depth-Aware
    Deep Learning Framework for Pothole Severity Assessment.

    This prototype uses a U-Net model with a ResNet-34 encoder trained
    on the Pothole440 dataset. The current application performs
    image-based segmentation, connected-region counting, visualization,
    and provisional image-area categorization.
    """)


elif page == "Limitations":
    st.header("Research limitations")
    st.warning("""
    - Connected regions are estimates, not guaranteed individual potholes.
    - Small regions may be filtered, and one pothole may be split into several.
    - Water reflections, shadows, poor lighting, and road texture can cause errors.
    - Image-area fraction is not physical surface area.
    - Calibrated depth, volume, validated engineering severity grades,
      reflection-specific robustness testing, and video tracking are not
      implemented in this version.
    """)
    st.write(
        "Predictions require human review and must not be treated as "
        "certified road-safety measurements."
    )

st.divider()
st.caption("PotholeAI Research Prototype | Computer Vision for Road Inspection")
