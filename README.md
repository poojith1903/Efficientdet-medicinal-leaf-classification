# Efficientdet-medicinal-leaf-classification
# 🌿 EfficientDet-Based Medicinal Leaf Classification

A deep learning framework for accurate classification of Indian medicinal leaves using the EfficientDet model with Grad-CAM visualizations and bounding box detection.

---

## 📌 Abstract

Accurate identification of Indian medicinal plants is vital for preserving traditional knowledge and enhancing healthcare. This project presents an EfficientDet-based system that classifies medicinal leaves and provides therapeutic usage insights. The model achieves **97% accuracy** on a dataset of **12,813 images from 98 species**, integrating **Grad-CAM** for interpretability and **bounding boxes** for localization.

---

## 📁 Dataset

- **Source**: [Indian Medicinal Leaves Dataset - Kaggle](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)
- **Classes**: 98 Medicinal Leaf Species
- **Images**: 12,813 high-resolution images

> 📌 Note: Due to Kaggle TOS, dataset is not included directly. [Download it here](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset) and place it in the `data/` folder.

---

## 🧠 Model Overview

- **Architecture**: EfficientDet-D0
- **Backbone**: EfficientNet-B0
- **Features**: BiFPN + Grad-CAM
- **Accuracy**: 97% on validation set

---

## 🧪 Key Features

- 🔍 **Multi-Class Leaf Classification**
- 🔬 **Grad-CAM Heatmap Visualization**
- 📦 **Bounding Box Localization**
- 💊 **Therapeutic Recommendation System**
- 🚀 **Gradio Interface for Real-Time Use**

---

## 🛠️ Installation

```bash
git clone https://github.com/poojith1903/Efficientdet-medicinal-leaf-classification
.git
cd Efficientdet-medicinal-leaf-classification

pip install -r requirements.txt
python app/gradio_app.ipynb
