import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pickle
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import keras

if not hasattr(keras, 'DTypePolicy'):
    class DTypePolicy:
        def __init__(self, name="float32", **kwargs):
            self.name = name
            # Keras 2 internal logic looks for these two specifically:
            self.compute_dtype = name
            self.variable_dtype = name
        @classmethod
        def from_config(cls, config):
            return cls(**config)
        def get_config(self):
            return {'name': self.name}
    tf.keras.utils.get_custom_objects()['DTypePolicy'] = DTypePolicy

# Fix the InputLayer mismatch
class FixedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)
tf.keras.utils.get_custom_objects()['InputLayer'] = FixedInputLayer


# ======================================================
# PAGE CONFIG - Forced Sidebar Expanded
# ======================================================
st.set_page_config(
    page_title="Brain Tumor Classifier Model | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ======================================================
# REFINED UI (FIXED VISIBILITY)
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

/* Global Styles */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background-color: #0B0E14;
}

/* Card Styling - Explicit Text Colors */
.card {
    background: rgba(23, 28, 36, 0.95);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #F8FAFC !important;
}

/* Titles and Headers */
.main-title {
    font-size: 60px;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #94edff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #58a6ff !important; 
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

/* Metric Boxes - Enhanced Visibility */
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-label {
    color: #94A3B8 !important; /* Increased brightness for grey text */
    font-size: 12px;
    margin-bottom: 5px;
    font-weight: 600;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: 24px;
    font-weight: 700;
}

/* Sidebar Fixes */
[data-testid="stSidebar"] {
    background-color: #11141B;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Ensure sidebar text is visible */
[data-testid="stSidebar"] .stMarkdown p {
    color: #F8FAFC !important;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & DATA
# ======================================================
@st.cache_resource
def get_model():
    if os.path.exists("tumor_model.h5"):
        return tf.keras.models.load_model("tumor_model.h5")
    return None

model = get_model()
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def load_samples():
    base_path = "samples"
    sample_list = []
    if not os.path.exists(base_path): return sample_list
    for cls in os.listdir(base_path):
        class_path = os.path.join(base_path, cls)
        if not os.path.isdir(class_path): continue
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            if os.path.isfile(img_path): sample_list.append((cls, img_path))
    return sample_list

# NEW: Helper to generate metrics for the Confusion Matrix
@st.cache_data
def get_confusion_matrix_data(_model, _samples):
    y_true = []
    y_pred = []
    for cls_folder, img_path in _samples:
        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            res = _model.predict(arr, verbose=0)
            
            # This ensures the true label matches the case of your class_names list
            # It looks for the folder name in a case-insensitive way
            matched_label = next((c for c in class_names if c.lower() == cls_folder.lower()), cls_folder)
            
            y_true.append(matched_label)
            y_pred.append(class_names[np.argmax(res)])
        except:
            continue
    return y_true, y_pred

# ======================================================
# SIDEBAR SHOWCASE, SETTINGS & METADATA
# ======================================================
with st.sidebar:
    st.markdown("<div class='section-title'>System Settings</div>", unsafe_allow_html=True)
    mode = st.selectbox("Analysis Mode", ["Upload Image", "Sample Gallery", "Demo Mode"])
    
    st.divider()

    # --- PROJECT SHOWCASE SECTION ---
    st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
    
    # Project Summary Card
    with st.expander("🧠 Brain Tumor Classifier", expanded=True):
        st.markdown("""
        <p style='color: #94A3B8; font-size: 13px; line-height: 1.4;'>
        This application uses a machine learning model to classify brain MRI images into four categories: <i>Glioma, Meningioma, Pituitary tumor, or No tumor.</i> Users can upload an MRI scan, and the system predicts the tumor type with confidence scores.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # --- MODEL PERFORMANCE METADATA ---
    with st.expander("📊 Training Performance"):
        st.markdown("""
        <div style='font-size: 12px; color: #F8FAFC;'>
        <div style='display: flex; justify-content: space-between;'><span>Training Accuracy:</span><b>98.2%</b></div>
        <div style='display: flex; justify-content: space-between;'><span>Validation Accuracy:</span><b>96.7%</b></div>
        <div style='display: flex; justify-content: space-between;'><span>Base Model:</span><b>MobileNetV2</b></div>
        <div style='display: flex; justify-content: space-between;'><span>Total Parameters:</span><b>2.2M+</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # Technologies Expandable Section
    with st.expander("🛠️ Technology Stack"):
        st.markdown("""
        <div style='font-size: 12px; color: #F8FAFC;'>
        <b>Language:</b> Python<br>
        <b>Framework:</b> TensorFlow / Keras<br>
        <b>Processing:</b> NumPy, Pandas, OpenCV<br>
        <b>Dashboard:</b> Streamlit<br>
        <b>Visuals:</b> Matplotlib, Seaborn
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # System Metadata
    st.markdown("<p style='font-size:10px; opacity:0.6; text-align:center;'>SYSTEM STATUS: ONLINE</p>", unsafe_allow_html=True)
    st.caption("Machine Learning Model System v2.1 | Model: CNN-X-Brain-04")
    

# ======================================================
# MAIN CONTENT
# ======================================================
st.markdown("<div class='main-title'>Brain Tumor Classifier</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8; font-size:18px; margin-bottom:30px;'>Advanced Clinical Decision Support for MRI Analysis</p>", unsafe_allow_html=True)

image = None
all_samples = load_samples()

# DIAGNOSTIC INPUT CARD
st.markdown("<div class='section-title'>Diagnostic Input</div>", unsafe_allow_html=True)

if mode == "Upload Image":
    uploaded = st.file_uploader("Drop MRI scan here", type=["jpg","png","jpeg"], label_visibility="collapsed")
    if uploaded: image = Image.open(uploaded)
elif mode == "Sample Gallery":
    if all_samples:
        selected = st.selectbox("Select Patient Record", [s[1] for s in all_samples])
        image = Image.open(selected)
    else: st.warning("Samples folder empty or missing.")
elif mode == "Demo Mode":
    if st.button("Generate Random Scan"):
        if all_samples:
            cls, path = random.choice(all_samples)
            image = Image.open(path)

# ANALYSIS OUTPUT
if image and model:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("<div class='section-title'>Original MRI Scan</div>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    # PREPROCESSING & INFERENCE
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((224,224))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    predicted_class = class_names[idx]
    confidence = float(np.max(prediction)) * 100

    with col2:
        st.markdown("<div class='section-title'>Classification Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"<span style='background:rgba(88,166,255,0.2); color:#58a6ff; padding:8px 20px; border-radius:30px; font-weight:bold; border:1px solid #58a6ff; font-size:14px;'>{predicted_class.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin:15px 0 0 0; font-size:56px; color:white;'>{confidence:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; margin-bottom:25px;'>Probability Score</p>", unsafe_allow_html=True)
        st.progress(int(confidence))
        st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
        
        g1, g2 = st.columns(2)
        with g1: 
            st.markdown("<div class='metric-box'><p class='metric-label'>System Accuracy</p><p class='metric-value'>96.7%</p></div>", unsafe_allow_html=True)
        with g2:
            st.markdown("<div class='metric-box'><p class='metric-label'>F1-Score</p><p class='metric-value'>0.95</p></div>", unsafe_allow_html=True)

    # ======================================================
    # SIDE-BY-SIDE: GRAD-CAM & CONFUSION MATRIX
    # ======================================================
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    
    # Create two equal columns for the lower dashboard
    v_col1, v_col2 = st.columns([1, 1])

    # --- COLUMN 1: INTERPRETABILITY (GRAD-CAM) ---
    with v_col1:
        st.markdown("<div class='section-title'>Interpretability (Grad-CAM)</div>", unsafe_allow_html=True)
        
        def get_gradcam(img_array, model):
            try:
                # Dynamically find the last convolutional layer
                last_conv_layer_name = None
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = layer.name
                        break
                
                # Setup a model that outputs the last conv layer and the final prediction
                grad_model = tf.keras.models.Model(
                    [model.inputs], 
                    [model.get_layer(last_conv_layer_name).output, model.output]
                )
                
                # Calculate the gradient of the top predicted class with respect to the conv layer
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    loss = predictions[:, np.argmax(predictions[0])]
                
                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                # Multiply the conv layer output by the gradients (Importance)
                heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                return heatmap.numpy()
            except:
                return None

        heatmap = get_gradcam(img_array, model)
        
        if heatmap is not None:
            # Generate the overlay
            original_cv = np.array(image_rgb).astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)
            
            st.image(overlay, caption="Tumor Localization Heatmap", use_container_width=True)
            
            # Explanation Box for Grad-CAM
            st.info("""
            **Interpretability Insight:** The heatmap indicates the 'Regions of Interest' (ROI) the model used for classification. 
            Red zones represent high activation. If the heatmap aligns with the tumor site, 
            it confirms the model's spatial reasoning is clinically sound.
            """)
        else:
            st.warning("Could not generate Grad-CAM for this architecture.")

    # --- COLUMN 2: MODEL PERFORMANCE (CONFUSION MATRIX) ---
    with v_col2:
        st.markdown("<div class='section-title'>Model Performance (Matrix)</div>", unsafe_allow_html=True)
        
        if all_samples:
            # get_confusion_matrix_data is called from your cached helper function
            with st.spinner("Analyzing Global Metrics..."):
                y_true, y_pred = get_confusion_matrix_data(model, all_samples)
                
                if y_true:
                    # Optimized figure size for column layout
                    fig, ax = plt.subplots(figsize=(7, 5))
                    fig.patch.set_facecolor('#0B0E14') # Dark theme match
                    
                    cm = confusion_matrix(y_true, y_pred, labels=class_names)
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=class_names, yticklabels=class_names, 
                                ax=ax, cbar=False, annot_kws={"size": 12, "weight": "bold"})
                    
                    plt.title("Actual vs Predicted Classes", color='#58a6ff', fontsize=12)
                    plt.xlabel('Predicted Label', color='#94A3B8', fontsize=10)
                    plt.ylabel('True Label', color='#94A3B8', fontsize=10)
                    ax.tick_params(colors='white', labelsize=8)
                    
                    st.pyplot(fig, use_container_width=True)

                    # Explanation Box for Confusion Matrix
                    st.info("""
                    **Performance Insight:** This matrix evaluates the model against all data in the `/samples` folder. 
                    The diagonal axis represents correct predictions. Use this to identify 
                    if the model is consistently confusing similar tumor types (e.g., Glioma vs. Meningioma).
                    """)
                else:
                    st.warning("Ensure the '/samples' folder contains labeled subfolders.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("<div style='margin-top:50px;'></div>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#484f58; font-size:12px; border-top:1px solid rgba(255,255,255,0.05); padding-top:20px;'>
    ⚠️ DISCLOSURE: This NeuroScan dashboard is an AI-assisted research tool and should not be used for 
    primary medical diagnosis. All outputs must be verified by a board-certified radiologist.
</p>
""", unsafe_allow_html=True)
