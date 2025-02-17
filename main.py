import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2  # OpenCV for contour extraction
from io import BytesIO
import zipfile
import plotly.graph_objects as go

# --------------------------
# Set Page Configuration
# --------------------------
st.set_page_config(page_title="MRI Classification, Segmentation, & 3D Visualization", page_icon="🌍", layout="wide")

# --------------------------
# Custom CSS for Styling
# --------------------------
st.markdown(
    """
    <style>
    body { background-color: #ffffff; color: black; }
    .sticky-header {
        position: sticky; top: 0; z-index: 1000;
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 20px; background-color: white;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sticky-header .logo { font-size: 24px; font-weight: bold; color: #007bff; }
    .sticky-header .header-buttons { display: flex; gap: 10px; }
    .sticky-header .header-buttons button {
        background-color: #007bff; color: white; font-size: 16px;
        padding: 8px 16px; border-radius: 8px; border: none; cursor: pointer;
    }
    .sticky-header .header-buttons button:hover { background-color: #0056b3; }
    .main-content { margin: 20px 0; display: flex; justify-content: space-between; align-items: center; }
    .main-content h1 { font-size: 32px; margin: 0; }
    .upload-button {
        background-color: #007bff; color: white; font-size: 16px;
        padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer;
    }
    .upload-button:hover { background-color: #0056b3; }
    .content { background-color: #C3E5FF; }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Sticky Header HTML
# --------------------------
st.markdown(
    """
    <div class="sticky-header">
        <div class="logo">Logo</div>
        <div class="header-buttons">
            <button>Home</button>
            <button>About</button>
            <button>Contact</button>
            <button>Settings</button>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="main-content">
        <h1>AI Tumor Detector</h1>
        <button class="upload-button">Upload Scan</button>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="content">
        <ul>
            <a class="content-1"><li>About</li></a>
            <a class="content-2"><li>Contact</li></a>
            <a class="content-3"><li>Other Info</li></a>
            <a class="content-4"><li>More</li></a>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================
# Custom Metric Functions
# ============================
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

tf.keras.utils.get_custom_objects()["dice_coef"] = dice_coef
tf.keras.utils.get_custom_objects()["iou_coef"] = iou_coef

# ============================
# Model Loading Functions (Cached)
# ============================
@st.cache_resource
def load_classification_model():
    return load_model("Models/classifier_model1.h5", custom_objects={"dice_coef": dice_coef, "iou_coef": iou_coef})

@st.cache_resource
def load_segmentation_model():
    return load_model("Models/segmentation_model.keras", custom_objects={"dice_coef": dice_coef, "iou_coef": iou_coef})

# ============================
# Preprocessing & Helper Functions
# ============================
def preprocess_image(image: Image.Image, target_size=(256, 256)) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def threshold_mask(mask: np.ndarray, threshold=0.5) -> np.ndarray:
    return (mask > threshold).astype(np.uint8)

def overlay_outline(base_img: Image.Image, mask: np.ndarray, outline_color=(0, 0, 255), thickness=1) -> Image.Image:
    """
    Draw the outline of the binary mask on the original image.
    """
    base_arr = np.array(base_img.convert("RGB"))
    base_bgr = cv2.cvtColor(base_arr, cv2.COLOR_RGB2BGR)
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base_bgr, contours, -1, outline_color, int(round(thickness)))
    overlay_arr = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_arr)

def pil_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def plot_slices(volume, title="Volume Slices"):
    """
    Create a Plotly figure that lets you slide through slices of a 3D volume.
    
    Parameters:
      volume (numpy.ndarray): 3D array of shape (num_slices, height, width)
      title (str): Figure title.
      
    Returns:
      Plotly figure with an interactive slider.
    """
    num_slices = volume.shape[0]
    # Create frames for each slice
    frames = [
        go.Frame(data=[go.Image(z=volume[i])], name=str(i))
        for i in range(num_slices)
    ]
    # Initial slice is the first one
    fig = go.Figure(
        data=go.Image(z=volume[0]),
        layout=go.Layout(
            title=title,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "y": 1,
                    "x": 1.12,
                    "xanchor": "right",
                    "yanchor": "top",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        }
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "method": "animate",
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": str(i),
                        }
                        for i in range(num_slices)
                    ],
                    "active": 0,
                    "currentvalue": {
                        "font": {"size": 12},
                        "prefix": "Slice: ",
                        "visible": True,
                        "xanchor": "center",
                    },
                    "pad": {"b": 10, "t": 50},
                }
            ],
        ),
    )
    fig.frames = frames
    return fig
# ============================
# Single Scan App
# ============================
def single_scan_app():
    st.header("Single Scan Segmentation")
    uploaded_file = st.file_uploader("Upload an MRI scan (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif"], key="single_scan")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)
        pred = clf_model.predict(img_array)
        score = float(np.squeeze(pred).item())
        st.write(f"Classification Probability (Flair): {score:.3f}")
        if score > 0.5:
            st.success("Lesion/Flair Detected. Performing segmentation...")
            with st.expander("Advanced Options", expanded=False):
                thresh = st.slider("Segmentation Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="seg_thresh_slider")
                thickness_value = st.slider("Outline Thickness", min_value=1.0, max_value=2.0, value=1.0, step=0.01, key="outline_thickness_slider")
                thickness = int(round(thickness_value))
                outline_option = st.selectbox("Outline Color", options=["Red", "Green"], key="outline_color_select")
                outline_color = (0, 0, 255) if outline_option == "Red" else (0, 255, 0)
            seg_pred = seg_model.predict(img_array)
            seg_mask = seg_pred[0]
            binary_mask = threshold_mask(seg_mask, threshold=thresh).squeeze()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Binary Mask")
                st.image(binary_mask * 255, use_column_width=True)
            with col3:
                st.subheader("Outline Overlay")
                outline_img = overlay_outline(image, binary_mask, outline_color=outline_color, thickness=thickness)
                st.image(outline_img, use_column_width=True)
            st.download_button(label="Download Segmentation Mask", data=pil_to_bytes(Image.fromarray((binary_mask * 255).astype(np.uint8))), file_name="segmentation_mask.png", mime="image/png")
        else:
            st.warning("No Flair/Lesion detected. Segmentation skipped.")

# ============================
# 3D Data Plotter App
# ============================
def plot_3d_data_app():
    st.header("3D Data Plotter")
    uploaded_zip = st.file_uploader("Upload ZIP of MRI scans (alternating MRI and segmentation images)", type=["zip"], key="zip_uploader")
    if uploaded_zip is not None:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            file_list = sorted([f for f in z.namelist() if not f.endswith("/")])
            if len(file_list) % 2 != 0:
                st.error("Expected an even number of files (alternating MRI and segmentation).")
                return
            mri_images = []
            seg_images = []
            for i, fname in enumerate(file_list):
                with z.open(fname) as f:
                    # Convert each image to grayscale.
                    img = Image.open(f).convert("L")
                    if i % 2 == 0:
                        mri_images.append(np.array(img))
                    else:
                        seg_images.append(np.array(img))
            if len(mri_images) == 0 or len(seg_images) == 0:
                st.error("No images found in ZIP file.")
                return
            mri_volume = np.stack(mri_images, axis=0)
            seg_volume = np.stack(seg_images, axis=0)
            st.write("MRI Volume shape:", mri_volume.shape)
            st.write("Segmentation Volume shape:", seg_volume.shape)
            st.write("MRI Intensity Range:", np.min(mri_volume), np.max(mri_volume))
            st.write("Segmentation Intensity Range:", np.min(seg_volume), np.max(seg_volume))
            
            # Create interactive slice plots
            fig_mri = plot_slices(mri_volume, title="MRI Slices")
            st.plotly_chart(fig_mri, use_container_width=True)
            
            fig_seg = plot_slices(seg_volume, title="Segmentation Slices")
            st.plotly_chart(fig_seg, use_container_width=True)

# ============================
# Main App with Tabs
# ============================
def main():
    st.title("MRI Classification, Segmentation, & 3D Visualization")
    tab1, tab2 = st.tabs(["Single Scan", "3D Data Plotter"])
    with tab1:
        single_scan_app()
    with tab2:
        plot_3d_data_app()

# Load global models once
clf_model = load_classification_model()
seg_model = load_segmentation_model()

if __name__ == "__main__":
    main()