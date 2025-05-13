import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input,
    Conv2D, UpSampling2D, Add
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# === STEP 1: Prepare Combined Dataset ===
leaf_src = r'C:\Users\LENOVO\Downloads\capstone project\Medicinal Leaf dataset'
plant_src = r'C:\Users\LENOVO\Downloads\capstone project\Medicinal plant dataset'
combined_dir = r'C:\Users\LENOVO\Downloads\capstone project\combined data'
os.makedirs(combined_dir, exist_ok=True)

def copy_all_classes(src, dst):
    for folder in os.listdir(src):
        src_folder = os.path.join(src, folder)
        if os.path.isdir(src_folder):
            dst_folder = os.path.join(dst, folder.strip().lower().replace(" ", "_"))
            os.makedirs(dst_folder, exist_ok=True)
            for img in os.listdir(src_folder):
                full_path = os.path.join(src_folder, img)
                if os.path.isfile(full_path):
                    shutil.copy(full_path, os.path.join(dst_folder, img))

copy_all_classes(leaf_src, combined_dir)
copy_all_classes(plant_src, combined_dir)

# === STEP 2: Create Datasets ===
batch_size = 32
img_size = (512,512)

train_ds = tf.keras.utils.image_dataset_from_directory(
    combined_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    combined_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes.")

# === Prefetch for Performance ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)



