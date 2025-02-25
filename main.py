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
# Apply custom CSS for styling
st.markdown(
    """
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
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
        margin-left: 50px;
    }
    .main-content h1 {
        font-size: 32px;
        margin: 0;
    }
    .upload-button {
        background-color: #323D84;
        color: white;
        font-size: 20px;
        font-weight: 800;
        padding: 25px 35px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        margin-right: 50px;
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
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

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

st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

# Space after main content
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .content {
            background-color: transparent;
            box-shadow: 0px -4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }
        .content h2 {
            font-size: 28px;
            margin: 0;
            padding: 20px;
        }
        .content hr {
            width: 80%;
            margin-top: 0;
            margin-left: 20px;
        }
        .content h3 {
            color: #ffffff;
            font-size: 22px;
            font-weight: 600;
            background-color: #007bff;
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .content p {
            font-size: 16px;
            margin-left: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <section class="content">
        <div>
            <h3>Symptómy</h3>
            <hr>
            <p>Nádory mozgu môžu spôsobiť rôzne príznaky v závislosti od ich veľkosti a umiestnenia.<br>
            <br>
            <strong>Bolesť hlavy</strong>
            Pretrvávajúca bolesť hlavy, ktorá sa zhoršuje ráno alebo pri kašľaní, môže byť príznakom.<br>
            <strong>Záchvaty</strong>
            Náhle záchvaty, najmä u ľudí bez predchádzajúcej histórie, môžu signalizovať prítomnosť nádoru.<br>
            <strong>Problémy so zrakom alebo sluchom</strong>
            Zmeny v zraku alebo počutí môžu byť spôsobené nádorom, ktorý tlačí na nervy.<br>
            <strong>Zmeny osobnosti alebo kognície</strong>
            Strata pamäti, zmätok alebo zmeny v správaní sú tiež možné príznaky.<br>
            <br>
            Ak si všimnete tieto príznaky, je dôležité vyhľadať lekársku pomoc.</p>
        </div>
        <br>
        <div>
            <h3>Liečba</h3>
            <hr>
            <p>Liečba nádorov mozgu závisí od typu nádoru, jeho veľkosti, umiestnenia a štádia. Najčastejšie možnosti zahŕňajú chirurgické odstránenie nádoru, ak je to možné, čo môže byť doplnené o ožarovanie alebo chemoterapiu na zničenie zvyškov nádorových buniek. V niektorých prípadoch sa používajú cielené terapie alebo imunoterapia na zameranie sa na konkrétne nádorové bunky. Možnosti liečby sa líšia v závislosti od individuálnych potrieb pacienta, preto je dôležité konzultovať s odborníkmi na liečbu mozgových nádorov.</p>
        </div>
        <div>
            <h3>Prevencia nádorov mozgu</h3>
            <hr>
            <p>Prevencia nádorov mozgu nie je úplne možná, no existujú niektoré faktory, ktoré môžu ovplyvniť riziko ich vzniku. Medzi rizikové faktory patrí genetická predispozícia, ako aj rodinná história nádorov mozgu. V niektorých prípadoch môžu byť nádorové ochorenia spôsobené aj vystavením určitým chemikáliám alebo žiareniu. Aj keď nie je možné úplne predísť vzniku nádoru, udržiavanie zdravého životného štýlu, ako je vyvážená strava, pravidelný pohyb a minimalizácia vystavenia škodlivým látkam, môže pomôcť znížiť niektoré rizikové faktory.</p>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        .footer {
            background-color: #fffff;
            box-shadow: 0px -4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
            text-align: center;
            padding-bottom: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        <p>copyright & contact info</p>
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

def plot_color_slices(volume, title="Volume Slices"):
    """
    Create a Plotly figure with a slider to view each slice of a 4D volume (num_slices, height, width, 3).
    """
    num_slices = volume.shape[0]
    frames = []
    for i in range(num_slices):
        frames.append(
            go.Frame(
                data=[go.Image(z=volume[i])],
                name=str(i)
            )
        )
    fig = go.Figure(
        data=[go.Image(z=volume[0])],
        layout=go.Layout(
            title=title,
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 1.15,
                "y": 1,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0}}],
                }]
            }],
            sliders=[{
                "steps": [{
                    "method": "animate",
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                    "label": str(i)
                } for i in range(num_slices)],
                "active": 0,
                "currentvalue": {"prefix": "Slice: "},
                "pad": {"b": 10, "t": 50},
            }],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
        )
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
    uploaded_zip = st.file_uploader("Upload ZIP with images in strict order: MRI0, Mask0, MRI1, Mask1, etc.", 
                                    type=["zip"], key="zip_uploader")
    if uploaded_zip is not None:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            file_list = sorted([f for f in z.namelist() if not f.endswith("/")])
            # Must be an even number of files: [MRI0, Mask0, MRI1, Mask1, ...]
            if len(file_list) % 2 != 0:
                st.error("Expected an even number of files (MRI, mask, MRI, mask...).")
                return
            
            mri_images = []
            seg_images = []
            
            # Read images in the order: MRI, Mask, MRI, Mask...
            for i, fname in enumerate(file_list):
                with z.open(fname) as f:
                    # Keep color
                    img = Image.open(f).convert("RGB")
                    arr = np.array(img)  # shape (H, W, 3)
                    
                    # If i is even -> MRI; if i is odd -> mask
                    if i % 2 == 0:
                        mri_images.append(arr)
                    else:
                        seg_images.append(arr)
            
            # Print out all masks first
            st.subheader("All Masks in Order")
            for i, mask_arr in enumerate(seg_images):
                st.image(mask_arr, caption=f"Mask {i}", use_column_width=True)
            
            # Then print out all MRIs
            st.subheader("All MRI Scans in Order")
            for i, mri_arr in enumerate(mri_images):
                st.image(mri_arr, caption=f"MRI {i}", use_column_width=True)
            
            if not mri_images or not seg_images:
                st.error("No images found in the ZIP.")
                return
            
            # Build 3D volumes in slice order. 
            # We'll assume #slices = len(mri_images). 
            # shape = (num_slices, H, W, 3)
            mri_volume = np.stack(mri_images, axis=0)
            seg_volume = np.stack(seg_images, axis=0)
            
            st.write("MRI Volume shape:", mri_volume.shape)
            st.write("Mask Volume shape:", seg_volume.shape)
            
            # Convert each slice to grayscale for 3D plane slicing
            # shape after conversion: (num_slices, H, W)
            mri_gray_list = []
            seg_gray_list = []
            for i in range(mri_volume.shape[0]):
                mri_gray_list.append(rgb_to_grayscale(mri_volume[i]))
                seg_gray_list.append(rgb_to_grayscale(seg_volume[i]))
            mri_gray_3d = np.stack(mri_gray_list, axis=0)  # shape (num_slices, H, W)
            seg_gray_3d = np.stack(seg_gray_list, axis=0)  # shape (num_slices, H, W)
            
            # Now create the 3D slicing scene
            fig_mri = plot_3d_slicing(mri_gray_3d, title="MRI in 3D Slicing")
            st.plotly_chart(fig_mri, use_container_width=True)
            
            fig_seg = plot_3d_slicing(seg_gray_3d, title="Mask in 3D Slicing")
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
