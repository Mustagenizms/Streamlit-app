import streamlit as st

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

def overlay_segmentation(base_img: Image.Image, mask: np.ndarray, alpha=0.5) -> Image.Image:
    """
    Overlay a binary mask onto the base image with adjustable transparency.
    """
    base_img = base_img.convert("RGB")
    overlay = Image.new("RGB", base_img.size, color=(255, 0, 0))  # red overlay
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(base_img.size)
    # Blend the base image and the overlay using the mask as alpha channel
    blended = Image.blend(base_img, overlay, alpha=alpha * np.array(mask_img).astype(np.float32)/255)
    return blended

# --------------------------
# 4. Main Streamlit App
# --------------------------
def main():
    st.title("MRI Classification & Segmentation")
    st.write("Upload an MRI to classify whether it has flair, adjust segmentation threshold, view an overlay, and download the mask.")

    # Load models (cached)
    clf_model = load_classification_model()
    seg_model = load_segmentation_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload an MRI scan (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        # Preprocess for classification
        img_array = preprocess_image(image)

        # Classification
        pred = clf_model.predict(img_array)
        score = float(np.squeeze(pred))
        st.write(f"Classification Probability (Flair): {score:.3f}")

        if score > 0.5:
            st.success("Lesion/Flair Detected. Performing segmentation...")

            # Allow user to adjust threshold and overlay transparency
            thresh = st.slider("Segmentation Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            alpha = st.slider("Overlay Transparency", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

            seg_pred = seg_model.predict(img_array)
            seg_mask = seg_pred[0]  # (H, W, 1)
            binary_mask = threshold_mask(seg_mask, threshold=thresh).squeeze()  # (H, W)

            # Display the binary mask
            st.image(binary_mask * 255, caption="Predicted Binary Mask", use_column_width=True)

            # Create and display an overlay
            overlay_img = overlay_segmentation(image, binary_mask, alpha=alpha)
            st.image(overlay_img, caption="Overlay of Segmentation on Original Image", use_column_width=True)

            # Provide a download link for the mask
            download_link = get_download_link(Image.fromarray((binary_mask * 255).astype(np.uint8)), "segmentation_mask.png")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.warning("No Flair/Lesion detected. Segmentation skipped.")

if __name__ == "__main__":
    main()