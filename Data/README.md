# 🌿 Indian Medicinal Leaves Dataset

A curated image dataset of Indian medicinal plant leaves, created to support research and development in machine learning, plant taxonomy, and traditional medicine informatics.

---

## 📖 Description

This dataset comprises high-quality images of various Indian medicinal leaves. The goal is to assist in developing AI models capable of identifying plant species from images and connecting them with known therapeutic uses.

---

## 📂 Dataset Structure

The dataset is organized into training and testing folders, with each subfolder named after the corresponding medicinal plant species.

indian_medicinal_leaves_dataset/
├── train/
│ ├── Neem/
│ ├── Aloe_Vera/
│ ├── Tulsi/
│ └── ...
├── test/
│ ├── Neem/
│ ├── Aloe_Vera/
│ ├── Tulsi/
│ └── ...


- **File Format**: JPEG (.jpg)
- **Color Mode**: RGB
- **Image Size**: Varies (typically resized to 224×224 or 512×512 during preprocessing)
- **Labels**: Derived from folder names
- **Classes**: 30+ medicinal leaf types
- **Total Images**: 10,000+

---

## 🧪 Applications

This dataset is ideal for:

- Training deep learning models for plant species classification
- Object detection tasks (with added annotations)
- Grad-CAM visualizations and explainability in model predictions
- Ethnobotanical AI systems for Ayurvedic usage recommendation
- Academic research and educational projects in machine learning and plant biology

---

## 🏷️ Sample Classes

| Class Name   | Scientific Name         | Common Uses                       |
|--------------|--------------------------|-----------------------------------|
| Neem         | *Azadirachta indica*     | Antiseptic, immunity booster      |
| Aloe Vera    | *Aloe barbadensis*       | Skin care, digestion aid          |
| Tulsi        | *Ocimum tenuiflorum*     | Respiratory relief, stress reducer |
| Bhringraj    | *Eclipta prostrata*      | Hair growth, liver detox          |
| Mint         | *Mentha arvensis*        | Cooling, digestive aid            |

*A full list of species and descriptions is available in the metadata file.*

---

## ⚙️ Usage Example (TensorFlow)

```python
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

