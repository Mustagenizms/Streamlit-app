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

# --------------------------
# 2. Load Models
# --------------------------
@st.cache_resource
def load_classification_model():
    """
    Loads and returns the classification model from the models/ folder.
    Adjust the path and add custom_objects if needed.
    """
    clf_model = load_model("models/classifier_model1.h5")  # or .keras
    return clf_model

@st.cache_resource
def load_segmentation_model():
    """
    Loads and returns the segmentation model from the models/ folder.
    Adjust the path and add custom_objects if needed.
    """
    seg_model = load_model("models/segmentation_model1.keras")  # or .keras
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

# --------------------------
# 4. Main Streamlit App
# --------------------------
def main():
    st.title("MRI Classification & Segmentation")
    st.write("Upload an MRI to classify whether it has flair, and if so, segment it.")

    # Load models (cached for performance)
    clf_model = load_classification_model()
    seg_model = load_segmentation_model()

    # --------------------------
    # File Uploader
    # --------------------------
    uploaded_file = st.file_uploader("Upload an MRI scan (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        # Display the uploaded MRI
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_column_width=True)

        # Preprocess for classification
        img_array = preprocess_image(image)  # shape (1, H, W, 3)

        # --------------------------
        # Classification
        # --------------------------
        pred = clf_model.predict(img_array)
        # Assume pred is shape (1,1) with a sigmoid output
        score = float(pred[0][0])  # Probability
        st.write(f"Classification Probability (Flair): {score:.3f}")

        # --------------------------
        # If Positive → Segmentation
        # --------------------------
        if score > 0.5:
            st.success("Lesion/Flair Detected. Performing segmentation...")
            seg_pred = seg_model.predict(img_array)  # shape (1, H, W, 1)
            seg_mask = seg_pred[0]  # shape (H, W, 1)
            
            # Threshold to get binary mask
            binary_mask = threshold_mask(seg_mask, threshold=0.5).squeeze()  # shape (H, W)

            # Display or overlay the mask
            st.write("Segmentation Mask:")
            st.image(binary_mask * 255, caption="Predicted Mask", use_column_width=True)

            # Potentially you could overlay it on the original image
            # Or provide a download button, etc.

        else:
            st.warning("No Flair/Lesion detected. Segmentation skipped.")

if __name__ == "__main__":
    main()