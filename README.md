# 🧠 Brain Tumor Classifier Model
### **Precision Medical Imaging & AI-Driven Diagnostics**

---

<div align="center">

<noautolink>
  <a href="https://brain-tumor-classifier-bypatilrajesh.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-DIRECT%20ACCESS-blue?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo Button" />
  </a>
</noautolink>

</div>

---

## 🚀 **Project Overview**
The **Brain Tumor Classifier Model** is a state-of-the-art deep learning solution designed to assist medical professionals in identifying and categorizing brain tumors from MRI scans. By combining a high-performance **Convolutional Neural Network (CNN)** with an intuitive, Figma-inspired interface, this tool bridges the gap between complex AI and clinical usability.

[Image of a professional medical imaging software interface showing MRI scans and classification results]

---

## ✨ **Key Features**

* **⚡ Real-Time Classification**: Instantly process MRI slices with high-speed inference.
* **🩺 Comprehensive Detection**: Classifies scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.
* **🔍 Explainable AI (Grad-CAM)**: Generates heatmaps to visualize the specific tumor regions the model identified, increasing diagnostic trust.
* **📊 Performance Dashboard**: View real-time evaluation metrics, including accuracy and loss curves.
* **🎨 Premium UI**: A dark-mode, glassmorphic interface optimized for clinical environments.

---

## 📊 **Model Evaluation Metrics**
The model has been rigorously validated to ensure clinical reliability and precision.

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Accuracy** | **96.7%** | ✅ |
| **Precision** | **95.3%** | ✅ |
| **Recall** | **94.8%** | ✅ |
| **F1-Score** | **95.0%** | ✅ |

[Image of a line graph showing the training and validation accuracy and loss over 20 epochs]

---

## 🛠️ **The Tech Stack**

* **Core Intelligence**: TensorFlow 2.x & Keras (CNN Architecture)
* **Computer Vision**: OpenCV & Pillow (PIL) for image preprocessing.
* **Frontend**: Streamlit for the high-fidelity UI.
* **Data Visualization**: Plotly for interactive diagnostic charts.
* **Deployment**: Streamlit Community Cloud.

---

## 🧩 **Explainable AI: How it Works**
The model utilizes **Grad-CAM** (Gradient-weighted Class Activation Mapping). This technique uses the gradients of the target class flowing into the final convolutional layer to produce a localization map highlighting the important regions in the image.

[Image of Grad-CAM heatmap overlay on a brain MRI scan highlighting tumor location]

---

## 📂 **Project Structure**
```text
├── app.py                # Main High-Fidelity Dashboard
├── tumor_model.h5        # Trained CNN Model weights
├── history.pkl           # Training performance logs
├── requirements.txt      # Deployment dependencies
├── samples/              # Curated MRI datasets for demo use
└── README.md             # Project Documentation
