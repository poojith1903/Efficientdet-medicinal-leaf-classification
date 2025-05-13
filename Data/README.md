# 🌿 Indian Medicinal Leaves Dataset

A curated image dataset of Indian medicinal plant leaves, created to support research and development in machine learning, plant taxonomy, and traditional medicine informatics.

---

## 📖 Description

This dataset comprises high-quality images of various Indian medicinal leaves. The goal is to assist in developing AI models capable of identifying plant species from images and connecting them with known therapeutic uses.


---

## 📦 Download the Dataset

You can download the dataset directly from Kaggle using the Kaggle CLI:

```bash
kaggle datasets download -d aryashah2k/indian-medicinal-leaves-dataset
unzip indian-medicinal-leaves-dataset.zip
**Kaggle link**: https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset

## 📂 Dataset Structure

The dataset is organized into training and testing folders, with each subfolder named after the corresponding medicinal plant species.

indian_medicinal_leaves_dataset/
├── train/
│ ├── Aloevera/
│ ├── Neem/
│ └── ...
├── test/
│ ├── Aloevera/
│ ├── Neem/
│ └── ...
├── medicinal_leaf_98_full_v19.csv


- **Image Format**: JPEG (.jpg)
- **Classes**: 98 medicinal leaf types
- **Metadata File**: `medicinal_leaf_98_full_v19.csv`

---

## 📋 Sample Metadata (from CSV)

| Common Name | Scientific Name         | Therapeutic Uses                                | Caution                              |
|-------------|--------------------------|--------------------------------------------------|---------------------------------------|
| Aloevera    | Aloe vera                | Soothes skin, promotes healing, digestive aid    | Excess may cause GI issues            |
| Amla        | Phyllanthus emblica      | Boosts immunity, aids digestion                  | May interact with blood thinners      |
| Guduchi     | Tinospora cordifolia     | Reduces fever, boosts immunity                   | Avoid during pregnancy                |
| Arali       | Nerium oleander          | Used for heart conditions (traditionally)        | **Highly toxic**, not for ingestion   |

*See the full list in [`medicinal_leaf_98_full_v19.csv`](./medicinal_leaf_98_full_v19.csv)*

---

## 🧪 Applications

- Medicinal plant classification using deep learning
- Object detection with models like EfficientDet
- Grad-CAM-based visual interpretability
- Therapeutic recommendation engines
- Educational tools in ethnobotany and Ayurveda

---

## ⚙️ Usage Example (EfficientDet)

```python
# Example: loading and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'indian_medicinal_leaves_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


val_generator = datagen.flow_from_directory(
    'indian_medicinal_leaves_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

