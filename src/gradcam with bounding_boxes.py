import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
from datetime import datetime

# Fix for the area display issue and improved GradCAM implementation

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM with a model and target layer
        """
        self.model = model
        self.layer_name = layer_name
        
        # If no layer is specified, use the last convolutional layer
        if self.layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    self.layer_name = layer.name
                    break
        
        # Create a model that maps input to both the activations and the predictions
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
    
    def _compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Generate a GradCAM heatmap for the specified class index
        """
        with tf.GradientTape() as tape:
            # Add batch dimension if needed
            if len(img_array.shape) == 3:
                img_array = tf.expand_dims(img_array, axis=0)
                
            # Cast to float32 for computation
            img_array = tf.cast(img_array, tf.float32)
            
            # Record operations for automatic differentiation
            tape.watch(img_array)
            
            # Get both the activations of the target layer and predictions
            conv_outputs, predictions = self.grad_model(img_array)
            
            # Determine prediction class if not specified
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Extract the class score for the specified class
            class_channel = predictions[:, class_idx]
        
        # This is the gradient of the class score with respect to the target layer outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool the gradients over all the axes except the channel axis
        # This is more stable than just taking the mean
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the output feature map with the gradients
        conv_outputs = conv_outputs[0]
        
        # Create the weighted feature map
        weighted_output = tf.multiply(pooled_grads, conv_outputs)
        
        # Average over channels to get the heatmap
        heatmap = tf.reduce_sum(weighted_output, axis=-1)
        
        # ReLU - only positive activations
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize the heatmap between 0 and 1
        max_val = tf.reduce_max(heatmap)
        if max_val != 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    def detect_leaf_bounding_box(self, img):
        # Convert to RGB if not already
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    
        if len(img.shape) != 3:
            # If grayscale, convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img.copy()
    
        # Convert to HSV color space for better color segmentation
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
        # Define range for green color (adjust these values based on your dataset)
        # These are sample values for green leaves
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
    
        # Create a mask for green regions
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        # Dilate to ensure we capture the entire leaf
        mask = cv2.dilate(mask, kernel, iterations=1)
    
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # If no contours found, try alternative method with Otsu thresholding
        if not contours:
        # Convert to LAB color space for better leaf/background separation
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
            # Extract the 'a' channel (green-red)
            a_channel = img_lab[:,:,1]
        
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
        
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Filter contours by area to avoid noise
        min_area = 100  # Minimum contour area to consider
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
        # If no valid contours found, return the whole image dimensions
        if not valid_contours:
            return (0, 0, img.shape[1], img.shape[0]), None
        # Get the largest contour (assuming it's the leaf)
        largest_contour = max(valid_contours, key=cv2.contourArea)
    
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        # Add a small margin to the bounding box
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
    
        return (x, y, w, h), largest_contour
    
    def add_text_with_background(self, img, text, position, font_scale=0.7, 
                                thickness=2, text_color=(255, 255, 255), 
                                bg_color=(0, 0, 0), padding=5):
        """
        Add text with background to an image
        """
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Calculate background rectangle
        x, y = position
        bg_rect = [
            x - padding, 
            y - text_height - padding - baseline, 
            x + text_width + padding, 
            y + padding
        ]
        
        # Draw background rectangle
        cv2.rectangle(img, 
                     (bg_rect[0], bg_rect[1]), 
                     (bg_rect[2], bg_rect[3]), 
                     bg_color, -1)
        
        # Draw text
        cv2.putText(img, text, (x, y - baseline), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   text_color, thickness)
        
        return img
    
    def generate_and_overlay_heatmap(self, img_path, class_idx=None, alpha=0.4):
        """
        Load an image, generate heatmap, and overlay it on the original image with bounding box
        """
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(512,512))
        img_array = image.img_to_array(img)
        
        # Process for model input
        model_input = tf.keras.applications.efficientnet.preprocess_input(img_array.copy())
        
        # Make prediction if class index is not provided
        preds = None
        if class_idx is None:
            preds = self.model.predict(np.expand_dims(model_input, axis=0))
            class_idx = np.argmax(preds[0])
            class_confidence = preds[0][class_idx]
            print(f"Predicted class: {class_names[class_idx]} with confidence: {class_confidence:.4f}")
        else:
            # Still need confidence for display
            preds = self.model.predict(np.expand_dims(model_input, axis=0))
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
        superimposed_img = cv2.addWeighted(img_bgr, 1-alpha, heatmap_colored, alpha, 0)
        
        # Detect leaf and get bounding box
        (x, y, w, h), contour = self.detect_leaf_bounding_box(img_array.astype('uint8'))
        
        # Draw bounding box on the original image copy
        img_with_bbox = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(img_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add a side panel for information
        class_name = class_names[class_idx]
        conf_text = f"{class_confidence:.2f}"
        
        # Calculate leaf metrics
        leaf_area = 0
        if contour is not None:
            leaf_area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * leaf_area / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            perimeter = 0
            circularity = 0
        
        # Side panel position
        panel_x = img_with_bbox.shape[1] - 230  # Right side of image
        panel_y = 30
        line_height = 30
        
        # Add info to side panel with fixed formatting - FIX for ?? in area
        labels = [
            f"Class: {class_name}",
            f"Conf: {conf_text}",
            f"Area: {int(leaf_area)}px",  # Format as integer to avoid ?? 
            f"Perim: {int(perimeter)}px",
            f"Circ: {circularity:.2f}"
        ]
        
        for i, label in enumerate(labels):
            img_with_bbox = self.add_text_with_background(
                img_with_bbox,
                label,
                (panel_x, panel_y + i * line_height),
                bg_color=(0, 100, 0)
            )
        
        # Draw the same side panel on superimposed image
        for i, label in enumerate(labels):
            superimposed_img = self.add_text_with_background(
                superimposed_img,
                label,
                (panel_x, panel_y + i * line_height),
                bg_color=(0, 100, 0)
            )
        
        # If contour is available, draw it
        if contour is not None:
            cv2.drawContours(superimposed_img, [contour], 0, (255, 0, 0), 2)
            cv2.drawContours(img_with_bbox, [contour], 0, (255, 0, 0), 2)
        
        # Convert back to RGB for display
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        img_with_bbox = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB)
        
        return img_array.astype('uint8'), heatmap, superimposed_img, img_with_bbox, class_idx, (x, y, w, h), class_confidence
    
    def plot_results(self, img_path, class_idx=None, save_dir=None, figsize=(12, 10)):
        """
        Plot the results in a 2x2 grid and optionally save the images
        """
        img, heatmap, overlaid_img, img_with_bbox, predicted_class, bbox, confidence = self.generate_and_overlay_heatmap(img_path, class_idx)
        
        # Create figure with 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot original image (top-left)
        axs[0, 0].imshow(img)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # Plot image with bounding box and class info (top-right)
        axs[0, 1].imshow(img_with_bbox)
        axs[0, 1].set_title('Leaf Detection & Classification')
        axs[0, 1].axis('off')
        
        # Plot heatmap (bottom-left)
        axs[1, 0].imshow(heatmap, cmap='jet')
        axs[1, 0].set_title('GradCAM Heatmap')
        axs[1, 0].axis('off')
        
        # Plot overlaid image with bounding box (bottom-right)
        axs[1, 1].imshow(overlaid_img)
        axs[1, 1].set_title(f'Prediction: {class_names[predicted_class]} ({confidence:.2f})')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create filename with timestamp and class name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.basename(img_path).split('.')[0]
            class_name = class_names[predicted_class].replace(' ', '_')
            
            # Save combined figure only
            fig_path = os.path.join(save_dir, f"{base_filename}_{class_name}_{timestamp}_combined.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Combined figure saved to: {fig_path}")
        
        plt.show()
        
        return predicted_class, bbox, confidence

# Example usage:
def test_gradcam(model_path, test_image_path, save_dir=None):
    """
    Test GradCAM visualization on a sample image and optionally save results
    """
    # Load trained model
    model = load_model(model_path)
    
    # Initialize GradCAM - we'll use the final activation layer before the global pooling
    gradcam = GradCAM(model, layer_name="top_activation")
    
    # Visualize GradCAM results
    predicted_class, bbox, confidence = gradcam.plot_results(test_image_path, save_dir=save_dir)
    
    print(f"Predicted class: {class_names[predicted_class]} with confidence: {confidence:.4f}")
    print(f"Detected leaf bounding box: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
    
    if save_dir:
        print(f"Results saved to: {save_dir}")
    
    return predicted_class, bbox, confidence

# Create a function to batch process multiple test images
def process_test_images(model_path, test_image_folder, save_dir=None, num_samples=5):
    """
    Process multiple test images and display GradCAM results in 2x2 grid layout
    """
    model = load_model(model_path)
    gradcam = GradCAM(model, layer_name="top_activation")
    
    # Get list of images from the test folder
    image_files = []
    for class_folder in os.listdir(test_image_folder):
        class_path = os.path.join(test_image_folder, class_folder)
        if os.path.isdir(class_path):
            files = [os.path.join(class_path, f) for f in os.listdir(class_path)[:num_samples]]
            image_files.extend(files)
    
    # Process a limited number of images
    samples = min(len(image_files), num_samples)
    results = []
    
    for i in range(samples):
        print(f"\nProcessing image {i+1}/{samples}")
        predicted_class, bbox, confidence = gradcam.plot_results(image_files[i], save_dir=save_dir)
        results.append({
            'image_path': image_files[i],
            'predicted_class': class_names[predicted_class],
            'confidence': confidence,
            'bounding_box': bbox
        })
    
    return results

# Advanced function to analyze leaf characteristics
def analyze_leaf_characteristics(img_path, bbox=None):
    """
    Analyze leaf characteristics like shape, color and texture
    """
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # If no bounding box provided, detect one
    if bbox is None:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest_contour)
        else:
            bbox = (0, 0, img.shape[1], img.shape[0])
    
    x, y, w, h = bbox
    leaf_region = img[y:y+h, x:x+w]
    
    # Color analysis (average RGB)
    avg_color = np.mean(leaf_region, axis=(0, 1))
    
    # Shape analysis
    gray_leaf = cv2.cvtColor(leaf_region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray_leaf, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    leaf_area = np.sum(binary > 0)
    bbox_area = w * h
    filling_ratio = leaf_area / bbox_area
    
    # Contour-based measurements
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    else:
        perimeter = 0
        area = 0
        circularity = 0
    
    results = {
        'avg_color_rgb': avg_color,
        'leaf_area': leaf_area,
        'filling_ratio': filling_ratio,
        'perimeter': perimeter,
        'circularity': circularity
    }
    
    return results

# Function to predict and analyze a leaf image
def predict_and_analyze_leaf(model_path, img_path, save_dir=None):
    """
    Make a prediction, visualize with GradCAM, and analyze leaf characteristics
    """
    # Load the model
    model = load_model(model_path)
    
    # Initialize GradCAM
    gradcam = GradCAM(model, layer_name="top_activation")
    
    # Get prediction and bounding box, and save results if requested
    predicted_class, bbox, confidence = gradcam.plot_results(img_path, save_dir=save_dir)
    
    # Analyze leaf characteristics
    leaf_analysis = analyze_leaf_characteristics(img_path, bbox)
    
    print("\nLeaf Analysis Results:")
    print(f"Predicted class: {class_names[predicted_class]} with confidence: {confidence:.4f}")
    print(f"Average RGB color: R={leaf_analysis['avg_color_rgb'][0]:.1f}, G={leaf_analysis['avg_color_rgb'][1]:.1f}, B={leaf_analysis['avg_color_rgb'][2]:.1f}")
    print(f"Leaf area: {leaf_analysis['leaf_area']} pixels")
    print(f"Filling ratio: {leaf_analysis['filling_ratio']:.2f}")
    print(f"Circularity: {leaf_analysis['circularity']:.2f} (1.0 is a perfect circle)")
    
    return predicted_class, bbox, confidence, leaf_analysis

# Example usage:
# 1. Test GradCAM with side labels on a single image and save results
save_dir = r"C:\Users\LENOVO\Downloads\capstone project\det_output"
test_gradcam(r"C:\Users\LENOVO\Downloads\capstone project\efficientnet_bifpn_80class.keras", 
             r"C:\Users\LENOVO\Downloads\capstone project\Medicinal Leaf dataset\Tulsi\20190822_173642.jpg",save_dir)
