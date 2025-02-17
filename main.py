import streamlit as st
from io import BytesIO
import plotly.graph_objects as go
import zipfile 
import io
# Set the page configuration
st.set_page_config(page_title="AI Tumor Detector", page_icon="🌍", layout="wide")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: black;
    }
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: white;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sticky-header .logo {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    .sticky-header .header-buttons {
        display: flex;
        gap: 10px;
    }
    .sticky-header .header-buttons button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .sticky-header .header-buttons button:hover {
        background-color: #0056b3;
    }
    .main-content {
        margin: 20px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-content h1 {
        font-size: 32px;
        margin: 0;
    }
    .upload-button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .upload-button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sticky Header
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

# Space between header and main content
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Main content
st.markdown(
    """
    <div class="main-content">
        <h1>AI Tumor Detector</h1>
        <button class="upload-button">Upload Scan</button>
    </div>
    """,
    unsafe_allow_html=True
)

# Space after main content
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .content {
            background-color: #C3E5FF;
        }
        .content ul {
            
        }
    </style>
    """,
    unsafe_allow_html=True
)

# next content
st.markdown(
    """
    <div class="content">
        <ul>
            <a class="content-1"><li>About</li></a>
            <a class="content-2"><li>Contact</li></a>
            <a class="content-3"><li>shdbf</li></a>
            <a class="content-4"><li>srag</li></a>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric: 2 * (|X ∩ Y|) / (|X| + |Y|)
    Measures overlap between y_true and y_pred.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
@tf.keras.utils.register_keras_serializable()
def iou_coef(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union metric: |X ∩ Y| / (|X ∪ Y|)
    Another common overlap measure for segmentation.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

tf.keras.utils.get_custom_objects()["dice_coef"] = dice_coef
tf.keras.utils.get_custom_objects()["iou_coef"] = iou_coef

# --------------------------
# 2. Load Models
# --------------------------
@st.cache_resource
def load_classification_model():
    """
    Loads and returns the classification model from the models/ folder.
    """
    clf_model = load_model(
        "Models/classifier_model1.h5",
        custom_objects={"dice_coef": dice_coef, "iou_coef": iou_coef}
    )
    return clf_model

@st.cache_resource
def load_segmentation_model():
    """
    Loads and returns the segmentation model from the models/ folder.
    """
    seg_model = load_model(
        "Models/segmentation_model.keras",
        custom_objects={"dice_coef": dice_coef, "iou_coef": iou_coef}
    )
    return seg_model

# --------------------------
# 3. Preprocessing & Helper Functions
# --------------------------
def preprocess_image(image: Image.Image, target_size=(256, 256)) -> np.ndarray:
    """
    Convert PIL image to numpy array scaled [0, 1].
    Return shape: (1, H, W, 3) for model prediction.
    """
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch of 1
    return img_array

def threshold_mask(mask: np.ndarray, threshold=0.5) -> np.ndarray:
    """
    Convert model output probabilities to a binary mask.
    mask shape: (H, W, 1) or (H, W)
    """
    return (mask > threshold).astype(np.uint8)
def get_download_link(img: Image.Image, filename: str) -> str:
    from io import BytesIO
    import base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download {filename}</a>'
    
def pil_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

import cv2
import numpy as np
from PIL import Image

def overlay_outline(base_img: Image.Image, mask: np.ndarray, outline_color=(0, 255, 0), thickness=2) -> Image.Image:
    """
    Draws only the outline of the binary mask on top of the original image.
    
    Parameters:
        base_img (PIL.Image): The original MRI image.
        mask (np.ndarray): A binary mask of shape (H, W) where 1 indicates the tumor.
        outline_color (tuple): The color of the outline in BGR format. 
                               For red use (0, 0, 255); for green use (0, 255, 0).
        thickness (int): The thickness of the outline.
        
    Returns:
        PIL.Image: The original image with the outline drawn over it.
    """
    # Convert the base image to a NumPy array (RGB)
    base_array = np.array(base_img.convert("RGB"))
    
    # Convert the image to BGR (OpenCV uses BGR by default)
    base_bgr = cv2.cvtColor(base_array, cv2.COLOR_RGB2BGR)
    
    # Ensure mask is uint8 and convert mask values to 0 or 255
    mask_uint8 = (mask.astype(np.uint8)) * 255

    # Find contours (external contours only)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours (outline) on the BGR image
    cv2.drawContours(base_bgr, contours, -1, outline_color, thickness)
    
    # Convert back to RGB
    overlay_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)

# --------------------------
# 4. Main Streamlit App
# --------------------------
def main():
    st.title("MRI Classification & Segmentation")
    st.write("Upload an MRI to classify whether it has flair, adjust segmentation threshold and outline thickness under Advanced Options, view the results side-by-side, and download the mask.")

    # Load models (cached)
    clf_model = load_classification_model()
    seg_model = load_segmentation_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload an MRI scan (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    
        # Preprocess image for classification
        img_array = preprocess_image(image)

        # Classification
        pred = clf_model.predict(img_array)
        score = float(np.squeeze(pred).item())
        st.write(f"Classification Probability (Flair): {score:.3f}")

        if score > 0.5:
            st.success("Lesion/Flair Detected. Performing segmentation...")

            # Advanced options inside an expander.
            with st.expander("Advanced Options", expanded=False):
                thresh = st.slider("Segmentation Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="seg_thresh_slider")
                thickness_value = st.slider("Outline Thickness", min_value=1.0, max_value=2.0, value=1.0, step=0.01, key="outline_thickness_slider")
                # Convert to integer for drawing.
                thickness = int(round(thickness_value))
                outline_option = st.selectbox("Outline Color", options=["Red", "Green"], key="outline_color_select")
                outline_color = (0, 0, 255) if outline_option == "Red" else (0, 255, 0)

            # Segmentation
            seg_pred = seg_model.predict(img_array)
            seg_mask = seg_pred[0]  # shape (H, W, 1)
            binary_mask = threshold_mask(seg_mask, threshold=thresh).squeeze()  # shape (H, W)
            
            # Create columns for side-by-side display
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
            
            # Download button for binary mask
            mask_bytes = pil_to_bytes(Image.fromarray((binary_mask * 255).astype(np.uint8)))
            st.download_button(
                label="Download Segmentation Mask",
                data=mask_bytes,
                file_name="segmentation_mask.png",
                mime="image/png"
            )
        else:
            st.warning("No Flair/Lesion detected. Segmentation skipped.")


st.write("Upload a .zip file containing your MRI slices and segmentation masks in order (alternating between the MRI slice and its segmentation mask, from bottom to top).")

# Allow user to upload a zip file.
uploaded_zip = st.file_uploader("Upload folder as zip", type=["zip"])

if uploaded_zip is not None:
    # Read the zip file in memory.
    zip_bytes = uploaded_zip.read()
    zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes), 'r')
    
    # Get list of file names and sort them (adjust sorting as needed)
    file_list = sorted([f for f in zip_file.namelist() if not f.endswith('/')])
    
    mri_slices = []
    seg_slices = []
    
    for idx, file_name in enumerate(file_list):
        file_data = zip_file.read(file_name)
        img = Image.open(io.BytesIO(file_data))
        # Convert image to grayscale for consistency.
        img_gray = np.array(img.convert("L"))
        if idx % 2 == 0:
            mri_slices.append(img_gray)
        else:
            # For segmentation masks, apply a threshold to binarize.
            seg_mask = (np.array(img.convert("L")) > 127).astype(np.uint8)
            seg_slices.append(seg_mask)
    
    if len(mri_slices) == 0 or len(seg_slices) == 0:
        st.error("Ensure your zip file includes both MRI slices and corresponding segmentation masks in alternating order.")
    else:
        # Stack slices to create 3D volumes.
        volume_mri = np.stack(mri_slices, axis=0)
        volume_seg = np.stack(seg_slices, axis=0)
        
        # Create a 3D volume visualization using Plotly.
        fig = go.Figure()

        # MRI volume: rendered in grayscale with lower opacity.
        fig.add_trace(go.Volume(
            value=volume_mri,
            opacity=0.1,
            isomin=50,
            isomax=255,
            surface_count=20,
            colorscale='Gray',
            name='MRI Volume'
        ))

        # Segmentation volume: binary mask overlay in red.
        fig.add_trace(go.Volume(
            value=volume_seg,
            opacity=0.3,
            isomin=0.5,
            isomax=1,
            surface_count=1,
            colorscale=[[0, 'red'], [1, 'red']],
            name='Tumor Segmentation'
        ))

        fig.update_layout(
            title="3D Visualization of MRI Scan and Segmented Tumor",
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Slice'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()