import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Brain Tumor Classifier Model | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ======================================================
# REFINED UI CSS
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #0B0E14; }
.card {
    background: rgba(23, 28, 36, 0.95);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #F8FAFC !important;
}
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
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-label { color: #94A3B8 !important; font-size: 12px; margin-bottom: 5px; font-weight: 600; }
.metric-value { color: #FFFFFF !important; font-size: 24px; font-weight: 700; }
[data-testid="stSidebar"] { background-color: #11141B; border-right: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & DATA
# ======================================================
@st.cache_resource
def get_model():
    try:
        model = tf.keras.models.load_model("tumor_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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

@st.cache_data
def get_confusion_matrix_data(_model, _samples):
    y_true, y_pred = [], []
    for cls_folder, img_path in _samples:
        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            res = _model.predict(arr, verbose=0)
            matched_label = next((c for c in class_names if c.lower() == cls_folder.lower()), cls_folder)
            y_true.append(matched_label)
            y_pred.append(class_names[np.argmax(res)])
        except: continue
    return y_true, y_pred

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("<div class='section-title'>System Settings</div>", unsafe_allow_html=True)
    mode = st.selectbox("Analysis Mode", ["Upload Image", "Sample Gallery", "Demo Mode"])
    st.divider()
    
    st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
    with st.expander("🧠 Brain Tumor Classifier", expanded=True):
        st.write("MRI classification into 4 categories using Deep Learning.")
    
    with st.expander("📊 Performance"):
        st.markdown("**Accuracy:** 96.7%<br>**Model:** MobileNetV2", unsafe_allow_html=True)

# ======================================================
# MAIN CONTENT
# ======================================================
st.markdown("<div class='main-title'>Brain Tumor Classifier</div>", unsafe_allow_html=True)

image = None
all_samples = load_samples()

if mode == "Upload Image":
    uploaded = st.file_uploader("Drop MRI scan here", type=["jpg","png","jpeg"], label_visibility="collapsed")
    if uploaded: image = Image.open(uploaded)
elif mode == "Sample Gallery":
    if all_samples:
        selected = st.selectbox("Select Patient Record", [s[1] for s in all_samples])
        image = Image.open(selected)
elif mode == "Demo Mode":
    if st.button("Generate Random Scan") and all_samples:
        cls, path = random.choice(all_samples)
        image = Image.open(path)

if image and model:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(image, use_container_width=True, caption="Original MRI Scan")

    # Inference
    img_array = np.array(image.convert("RGB").resize((224,224)), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    conf = float(np.max(prediction)) * 100

    with col2:
        st.markdown(f"### Result: {class_names[idx]}")
        st.markdown(f"## {conf:.1f}% Confidence")
        st.progress(int(conf))
        
        g1, g2 = st.columns(2)
        g1.metric("Accuracy", "96.7%")
        g2.metric("F1-Score", "0.95")

    # --- Grad-CAM ---
    st.divider()
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("<div class='section-title'>Interpretability (Grad-CAM)</div>", unsafe_allow_html=True)
        try:
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break
            
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
            with tf.GradientTape() as tape:
                conv_outs, preds = grad_model(img_array)
                loss = preds[:, idx]
            
            grads = tape.gradient(loss, conv_outs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_outs[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
            
            original_cv = np.array(image.convert("RGB"))
            heatmap_res = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)
            st.image(overlay, use_container_width=True)
        except:
            st.warning("Grad-CAM visualization unavailable for this model structure.")

    with v_col2:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        if all_samples:
            y_true, y_pred = get_confusion_matrix_data(model, all_samples)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0B0E14')
            cm = confusion_matrix(y_true, y_pred, labels=class_names)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

st.markdown("<p style='text-align:center; opacity:0.5; font-size:10px;'>⚠️ Not for Medical Diagnosis</p>", unsafe_allow_html=True)
