# app.py
import os
import base64
import shutil
import subprocess
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# For metrics calculation
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'input')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'output')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_metrics():
    # Define file paths
    input_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask.png')
    output_img_path = os.path.join(app.config['OUTPUT_FOLDER'], 'inpainted_img.png')
    
    # Check that all necessary files exist
    if not os.path.exists(input_img_path) or not os.path.exists(mask_path) or not os.path.exists(output_img_path):
        return None
    
    resize_size = (512, 512)
    
    # Load images
    input_image = cv2.imread(input_img_path)
    output_image = cv2.imread(output_img_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    input_image = cv2.resize(input_image, resize_size)
    output_image = cv2.resize(output_image, resize_size)
    mask_image = cv2.resize(mask_image, resize_size)
    
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    # Create masked input (simulate what was fed into the model)
    _, binary_mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
    masked_input = input_image.copy()
    masked_input[binary_mask == 255] = [255, 255, 255]
    masked_input_rgb = cv2.cvtColor(masked_input, cv2.COLOR_BGR2RGB)
    
    # Global metrics
    input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(input_gray, output_gray)
    psnr_score = psnr(input_image_rgb, output_image_rgb)
    mse = np.mean((input_image_rgb.astype(np.float32) - output_image_rgb.astype(np.float32))**2)
    l1 = np.mean(np.abs(input_image_rgb.astype(np.float32) - output_image_rgb.astype(np.float32)))
    
    # Color difference in LAB space
    input_lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    output_lab = cv2.cvtColor(output_image, cv2.COLOR_BGR2LAB)
    color_diff = np.mean(np.sqrt(np.sum((input_lab.astype(np.float32) - output_lab.astype(np.float32))**2, axis=2)))
    
    metrics = {
        'global_ssim': round(ssim_score, 4),
        'global_psnr': round(psnr_score, 2),
        'global_mse': round(mse, 2),
        'global_l1': round(l1, 2),
        'global_color_diff': round(color_diff, 2)
    }
    
    # Metrics for the inpainted (damaged/restored) region
    inpaint_mask = binary_mask == 255
    if np.sum(inpaint_mask) > 0:
        masked_ssim = ssim(input_gray, output_gray, win_size=7, full=True)[1]
        masked_ssim_value = np.sum(masked_ssim * inpaint_mask) / np.sum(inpaint_mask)
        masked_mse = np.mean(((input_image_rgb[inpaint_mask] - output_image_rgb[inpaint_mask])**2))
        masked_l1 = np.mean(np.abs(input_image_rgb[inpaint_mask] - output_image_rgb[inpaint_mask]))
        masked_color_diff = np.mean(np.sqrt(np.sum((input_lab[inpaint_mask] - output_lab[inpaint_mask])**2, axis=1)))
        
        metrics.update({
            'region_ssim': round(masked_ssim_value, 4),
            'region_mse': round(masked_mse, 2),
            'region_l1': round(masked_l1, 2),
            'region_color_diff': round(masked_color_diff, 2)
        })
    else:
        metrics.update({
            'region_ssim': 'N/A',
            'region_mse': 'N/A',
            'region_l1': 'N/A',
            'region_color_diff': 'N/A'
        })
    
    return metrics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # --- Step 1: Handle input image upload ---
    if 'input_image' not in request.files:
        return "No input image provided", 400
    file = request.files['input_image']
    if file and allowed_file(file.filename):
        # Save input image as input_img.png
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
        file.save(input_path)
    else:
        return "Invalid input image", 400
    
    # --- Step 2: Handle mask selection ---
    mask_option = request.form.get('mask_option')
    
    if mask_option == 'upload_mask':
        mask_file = request.files.get('mask_image')
        if mask_file and allowed_file(mask_file.filename):
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask.png')
            mask_file.save(mask_path)
        else:
            return "Invalid mask file", 400
    elif mask_option == 'draw_mask':
        # Get the base64 image string from the canvas
        canvas_data = request.form.get('canvas_data')
        if canvas_data:
            header, encoded = canvas_data.split(',', 1)
            mask_data = base64.b64decode(encoded)
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask.png')
            with open(mask_path, 'wb') as f:
                f.write(mask_data)
        else:
            return "No canvas data provided", 400
    elif mask_option in ['random_freeform', 'circular', 'bbox', 'edge']:
        # Call your mask generation functions here (importing within the branch)
        if mask_option == 'random_freeform':
            import create_mask
            create_mask.create_ff_mask()
        elif mask_option == 'circular':
            import create_mask
            create_mask.create_circular_mask_with_ui()
        elif mask_option == 'bbox':
            import create_mask
            create_mask.create_bbox_mask()
        elif mask_option == 'edge':
            import edge_detection_mask
            edge_detection_mask.create_edge_mask(os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png'))
    else:
        return "Invalid mask option", 400

    # --- Step 3: Run the inpainting process ---
    try:
        # This will run your inpaint.py script which prints the status messages,
        # loads the model, and saves the output to the output folder.
        subprocess.run(['python', 'inpaint.py'], check=True)
    except subprocess.CalledProcessError as e:
        return f"Inpainting process failed: {e}", 500

    # --- Step 4: Calculate metrics ---
    metrics = calculate_metrics()
    
    # --- Step 5: Render results page ---
    return render_template('results.html', metrics=metrics)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
