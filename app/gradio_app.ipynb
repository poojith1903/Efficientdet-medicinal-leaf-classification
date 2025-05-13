import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# Load the model
model_path = r"C:\Users\LENOVO\Downloads\capstone project\efficientnet_bifpn_80class_final.keras"
model = load_model(model_path)

# Define class names (these should match your actual class names from training)
class_names = [
    'aloevera', 'amla', 'amruta_balli', 'amruthaballi', 'arali', 'ashoka', 
    'ashwagandha', 'astma_weed', 'avacado', 'badipala', 'balloon_vine', 'bamboo', 
    'basale', 'beans', 'betel', 'betel_nut', 'bhrami', 'brahmi', 'bringaraja', 
    'camphor', 'caricature', 'castor', 'catharanthus', 'chakte', 'chilly', 
    'citron_lime_(herelikai)', 'coffee', 'common_rue(naagdalli)', 'coriender', 'curry', 
    'curry_leaf', 'doddapatre', 'doddpathre', 'drumstick', 'ekka', 'eucalyptus', 
    'ganigale', 'ganike', 'gasagase', 'gauva', 'geranium', 'ginger', 'globe_amarnath', 
    'guava', 'henna', 'hibiscus', 'honge', 'insulin', 'jackfruit', 'jasmine', 
    'kamakasturi', 'kambajala', 'kasambruga', 'kepala', 'kohlrabi', 'lantana', 
    'lemon', 'lemon_grass', 'lemongrass', 'malabar_nut', 'malabar_spinach', 'mango', 
    'marigold', 'mint', 'nagadali', 'neem', 'nelavembu', 'nerale', 'nithyapushpa', 
    'nooni', 'onion', 'padri', 'palak(spinach)', 'papaya', 'pappaya', 'parijatha', 
    'pea', 'pepper', 'pomegranate', 'pomoegranate', 'pumpkin', 'raddish', 
    'raktachandini', 'rose', 'sampige', 'sapota', 'seethaashoka', 'seethapala', 
    'spinach1', 'tamarind', 'taro', 'tecoma', 'thumbe', 'tomato', 'tulasi', 'tulsi', 
    'turmeric', 'wood_sorel'
]

# Load therapeutic information
def load_therapeutic_info():
    # CSV file path - adjust according to your file location
    csv_path = r"C:\Users\LENOVO\Downloads\capstone project\medicinal_leaf_98_full_v19.csv"
    try:
        df = pd.read_csv(csv_path)
        # Create a dictionary with class name as key
        therapeutic_info = {}
        for _, row in df.iterrows():
            common_name = row.get('Common Name', '').lower().strip()
            if common_name in class_names:
                therapeutic_info[common_name] = {
                    'scientific_name': row.get('Scientific Name', 'N/A'),
                    'local_names': row.get('Alternate/Local Names', 'N/A'),
                    'therapeutic_uses': row.get('Therapeutic Uses', 'No information available'),
                    'preparation': row.get('Preparation', 'No information available'),
                    'caution': row.get('Caution', 'No specific cautions noted')
                }
        return therapeutic_info
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}

therapeutic_info = load_therapeutic_info()

# GradCAM implementation
class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        
        if self.layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.layer_name = layer.name
                    break
        
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
    
    def _compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        with tf.GradientTape() as tape:
            if len(img_array.shape) == 3:
                img_array = tf.expand_dims(img_array, axis=0)
                
            img_array = tf.cast(img_array, tf.float32)
            tape.watch(img_array)
            
            conv_outputs, predictions = self.grad_model(img_array)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_channel = predictions[:, class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        weighted_output = tf.multiply(pooled_grads, conv_outputs)
        
        heatmap = tf.reduce_sum(weighted_output, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        
        max_val = tf.reduce_max(heatmap)
        if max_val != 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    def detect_leaf_bounding_box(self, img):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    
        if len(img.shape) != 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img.copy()
    
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
    
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        mask = cv2.dilate(mask, kernel, iterations=1)
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if not contours:
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            a_channel = img_lab[:,:,1]
            blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        min_area = 100
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
        if not valid_contours:
            return (0, 0, img.shape[1], img.shape[0]), None
            
        largest_contour = max(valid_contours, key=cv2.contourArea)
    
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
    
        return (x, y, w, h), largest_contour
    
    def generate_overlay_heatmap(self, img_path_or_array):
        if isinstance(img_path_or_array, str):
            img = image.load_img(img_path_or_array, target_size=(512, 512))
            img_array = image.img_to_array(img)
        else:
            # Resize if necessary
            if img_path_or_array.shape[:2] != (512, 512):
                img_path_or_array = cv2.resize(img_path_or_array, (512, 512))
            img_array = img_path_or_array
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack((img_array,) * 3, axis=-1)
        
        # Process for model input
        model_input = tf.keras.applications.efficientnet.preprocess_input(img_array.copy())
        
        # Make prediction
        preds = self.model.predict(np.expand_dims(model_input, axis=0))
        class_idx = np.argmax(preds[0])
        class_confidence = preds[0][class_idx]
        
        # Compute heatmap
        heatmap = self._compute_heatmap(model_input, class_idx)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Convert heatmap to RGB format
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cm.jet(heatmap_colored)[:, :, :3] * 255
        heatmap_colored = np.uint8(heatmap_colored)
        
        # Convert original image to BGR (for OpenCV)
        img_bgr = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        
        # Detect leaf and get bounding box
        (x, y, w, h), contour = self.detect_leaf_bounding_box(img_array.astype('uint8'))
        
        # Draw bounding box on the superimposed image
        cv2.rectangle(superimposed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # If contour is available, draw it
        if contour is not None:
            cv2.drawContours(superimposed_img, [contour], 0, (255, 0, 0), 2)
        
        # Convert back to RGB for display
        result_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Calculate leaf metrics
        leaf_area = 0
        if contour is not None:
            leaf_area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * leaf_area / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            perimeter = 0
            circularity = 0
            
        leaf_metrics = {
            "area": int(leaf_area),
            "perimeter": int(perimeter),
            "circularity": round(circularity, 2)
        }
        
        return result_img, class_names[class_idx], class_confidence, leaf_metrics

# Function to predict and get therapeutic info
def predict_leaf(image):
    if image is None:
        return None, "Please upload an image."
    
    # Initialize GradCAM
    gradcam = GradCAM(model, layer_name="top_activation")
    
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Process image with GradCAM
    processed_image, class_name, confidence, leaf_metrics = gradcam.generate_overlay_heatmap(image_np)
    
    # Get therapeutic information
    info = therapeutic_info.get(class_name, {})
    scientific_name = info.get('scientific_name', 'Information not available')
    local_names = info.get('local_names', 'Information not available')
    therapeutic_uses = info.get('therapeutic_uses', 'Information not available')
    preparation = info.get('preparation', 'Information not available')
    caution = info.get('caution', 'Information not available')
    
    # Format the combined results HTML
    combined_html = f"""
    <div class="results-container">
        <div class="results-section">
            <div class="identification-card">
                <div class="card-header">
                    <h3>Identification Results</h3>
                </div>
                <div class="card-content">
                    <div class="result-item">
                        <span class="label">Identified Plant:</span>
                        <span class="value">{class_name.replace('_', ' ').title()}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Confidence:</span>
                        <span class="value">{confidence:.2%}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%;"></div>
                        </div>
                    </div>
                    <div class="result-item">
                        <span class="label">Scientific Name:</span>
                        <span class="value italic">{scientific_name}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Local Names:</span>
                        <span class="value">{local_names}</span>
                    </div>
                    
                    <div class="metrics-section">
                        <h4>Leaf Metrics</h4>
                        <div class="metrics-grid">
                            <div class="metric-box">
                                <div class="metric-value">{leaf_metrics['area']}</div>
                                <div class="metric-label">Area (px²)</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{leaf_metrics['perimeter']}</div>
                                <div class="metric-label">Perimeter (px)</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-value">{leaf_metrics['circularity']}</div>
                                <div class="metric-label">Circularity</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="results-section">
            <div class="therapeutic-card">
                <div class="card-header">
                    <h3>Therapeutic Information</h3>
                </div>
                <div class="card-content">
                    <div class="therapeutic-section">
                        <h4>Therapeutic Uses</h4>
                        <p class="therapeutic-text">{therapeutic_uses}</p>
                    </div>
                    <div class="therapeutic-section">
                        <h4>Preparation Methods</h4>
                        <p class="therapeutic-text">{preparation}</p>
                    </div>
                    <div class="caution-box">
                        <div class="caution-header">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                            <h4>Cautions</h4>
                        </div>
                        <p class="caution-text">{caution}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    return processed_image, combined_html

# Custom CSS for the unified interface
custom_css = """
:root {
    --primary: #2D7D46;
    --primary-light: #E3F2E7;
    --accent: #FFC107;
    --text-dark: #2C3E50;
    --text-light: #7F8C8D;
    --text-therapeutic: #333333;  /* Darker text for better visibility */
    --danger: #E74C3C;
    --danger-light: #FDEDEC;
    --radius: 12px;
    --shadow: 0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1);
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

.app-header {
    background: linear-gradient(135deg, var(--primary), #1e5631);
    padding: 1.5rem;
    border-radius: var(--radius) var(--radius) 0 0;
    position: relative;
    overflow: hidden;
}

.app-header h1 {
    margin: 0;
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.app-header p {
    color: rgba(255, 255, 255, 0.8);
    margin: 0.5rem 0 0;
    font-size: 1rem;
    max-width: 80%;
}

.header-decoration {
    position: absolute;
    right: -50px;
    top: -20px;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle at center, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
}

.app-footer {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0 0 var(--radius) var(--radius);
    text-align: center;
    font-size: 0.85rem;
    color: var(--text-light);
    border-top: 1px solid #eaeaea;
}

.card {
    background: white;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    height: 100%;
}

.upload-container {
    background: white;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    height: 100%;
    padding: 0;
    /* Reduced height for upload box */
    max-height: 300px;
}

.card-header {
    padding: 1rem;
    background-color: var(--primary-light);
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.card-header h3 {
    margin: 0;
    color: var(--primary);
    font-size: 1.2rem;
    font-weight: 600;
}

.card-content {
    padding: 1rem;
}

.upload-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem;  /* Reduced padding */
    background-color: #f9fafc;
    border: 2px dashed #d1d5db;
    border-radius: var(--radius);
    cursor: pointer;
    transition: all 0.2s ease;
    /* Reduced height */
    max-height: 200px;
}

.upload-section:hover {
    border-color: var(--primary);
    background-color: var(--primary-light);
}

.upload-icon {
    font-size: 2rem;  /* Smaller icon */
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.analyze-btn {
    width: 100%;
    padding: 0.75rem;
    margin-top: 1rem;
    background-color: var(--primary);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    transition: all 0.2s ease;
}

.analyze-btn:hover {
    background-color: #236639;
    transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}

.analyze-btn:active {
    transform: translateY(0);
}

.result-image {
    width: 100%;
    border-radius: var(--radius);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.identification-card, .therapeutic-card {
    border-radius: var(--radius);
    background: white;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.identification-card .card-header {
    background-color: #e3f2fd;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.identification-card .card-header h3 {
    color: #1976d2;
}

.therapeutic-card .card-header {
    background-color: #e8f5e9;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}

.therapeutic-card .card-header h3 {
    color: var(--primary);
}

.result-item {
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #f0f0f0;
}

.result-item:last-child {
    border-bottom: none;
}

.label {
    display: block;
    font-size: 0.85rem;
    color: var(--text-light);
    margin-bottom: 0.25rem;
}

.value {
    font-weight: 500;
    color: var(--text-dark);
}

.italic {
    font-style: italic;
}

.confidence-bar {
    height: 6px;
    background-color: #eaeaea;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.5rem;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #4caf50, #8bc34a);
    border-radius: 3px;
}

.metrics-section {
    margin-top: 1.5rem;
}

.metrics-section h4 {
    margin: 0 0 1rem;
    color: var(--text-dark);
    font-size: 1rem;
    font-weight: 600;
}

.metrics-grid {
    display: flex;
    gap: 1rem;
}

.metric-box {
    flex: 1;
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 0.25rem;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-light);
}

.therapeutic-section {
    margin-bottom: 1.5rem;
}

.therapeutic-section h4 {
    margin: 0 0 0.75rem;
    font-size: 1rem;
    color: var(--primary);
}

/* Improved text visibility for therapeutic information */
.therapeutic-text {
    color: var(--text-therapeutic);
    font-weight: 500;
    line-height: 1.6;
}

.caution-box {
    background-color: var(--danger-light);
    border-radius: 8px;
    padding: 1rem;
}

.caution-header {
    display: flex;
    align-items: center;
    margin-bottom: 0.75rem;
}

.caution-header svg {
    color: var(--danger);
    margin-right: 0.5rem;
}

.caution-header h4 {
    margin: 0;
    color: var(--danger);
}

/* Improved text visibility for caution information */
.caution-text {
    color: #5A231F;
    font-weight: 500;
}

.loading-animation {
    display: none;
    text-align: center;
    padding: 2rem;
}

.loading-animation.active {
    display: block;
}

.spinner {
    width: 40px;
    height: 40px;
    margin: 0 auto 1rem;
    border: 4px solid rgba(0,0,0,0.1);
    border-radius: 50%;
    border-left-color: var(--primary);
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* New combined results container */
.results-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.results-section {
    width: 100%;
}

/* Responsive styling */
@media (min-width: 768px) {
    .results-container {
        flex-direction: row;
        flex-wrap: wrap;
    }
    
    .results-section {
        flex: 1;
        min-width: 300px;
    }
}

@media (max-width: 768px) {
    .metrics-grid {
        flex-direction: column;
    }
    
    .app-header p {
        max-width: 100%;
    }
    
    .header-decoration {
        display: none;
    }
}
"""

# Create Gradio interface with the unified design
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
    <div class="app-header">
        <div class="header-decoration"></div>
        <h1>Medicinal Leaf Analyzer</h1>
        <p>Upload an image of a medicinal leaf to identify it and learn about its therapeutic properties</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Smaller upload container
            with gr.Column(elem_classes=["upload-container"]):
                input_image = gr.Image(
                    label="Upload Leaf Image", 
                    type="numpy",
                    elem_classes=["upload-image"],
                    height=200  # Reduced height
                )
                submit_btn = gr.Button(
                    "Analyze Leaf", 
                    variant="primary",
                    elem_classes=["analyze-btn"]
                )
        
        with gr.Column(scale=2):
            # Output image for visualization
            output_image = gr.Image(
                label="Analysis Visualization",
                elem_classes=["result-image"]
            )
    
    # Combined outputs for identification and therapeutic info
    combined_results = gr.HTML(
        elem_id="combined-results"
    )
    
    gr.HTML("""
    <div class="app-footer">
        <p>Medicinal Leaf Analysis System | Powered by EfficientNet with BiFPN</p>
        <p><small>© 2025 - Plant Identification and Therapeutic Information System</small></p>
    </div>
    """)
    
    # Add loading animation with JavaScript
    gr.HTML("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const analyzeBtn = document.querySelector('.analyze-btn');
        
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', function() {
                analyzeBtn.textContent = 'Analyzing...';
                analyzeBtn.disabled = true;
                
                setTimeout(function() {
                    analyzeBtn.textContent = 'Analyze Leaf';
                    analyzeBtn.disabled = false;
                }, 100);
            });
        }
    });
    </script>
    """)
    
    submit_btn.click(
        fn=predict_leaf, 
        inputs=[input_image], 
        outputs=[output_image, combined_results]
    )
    
    demo.launch(share=True)
