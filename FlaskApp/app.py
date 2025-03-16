# app.py
import os
import base64
import shutil
import subprocess
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, make_response, Response
from werkzeug.utils import secure_filename
import queue
import threading

# For metrics calculation
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add imports for pattern_continuity, edge_coherence, and texture_consistency
from skimage.feature import local_binary_pattern

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'input')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'output')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add these metric functions before your calculate_metrics() function
def pattern_continuity(orig_img, gen_img, mask, kernel_size=5):
    # Convert to grayscale
    if len(orig_img.shape) > 2:
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = orig_img
        gen_gray = gen_img

    # Create filters at different orientations
    orientations = 8
    gabor_responses_orig = []
    gabor_responses_gen = []

    for i in range(orientations):
        theta = np.pi * i / orientations
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)

        filtered_orig = cv2.filter2D(orig_gray, cv2.CV_8UC3, kernel)
        filtered_gen = cv2.filter2D(gen_gray, cv2.CV_8UC3, kernel)

        # Dilate mask to include border region
        kernel_dilate = np.ones((3, 3), np.uint8)
        mask_border = cv2.dilate(mask, kernel_dilate, iterations=1)

        # Get responses at the border of the inpainted region
        border_region = (mask_border > 0) & (mask == 0)

        if np.sum(border_region) > 0:
            gabor_responses_orig.append(filtered_orig[border_region])
            gabor_responses_gen.append(filtered_gen[border_region])

    # Calculate correlation between original and generated responses
    if len(gabor_responses_orig) == 0:
        return 0

    correlations = []
    for orig_resp, gen_resp in zip(gabor_responses_orig, gabor_responses_gen):
        if len(orig_resp) > 0 and len(gen_resp) > 0:
            corr = np.corrcoef(orig_resp.flatten(), gen_resp.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0

def edge_coherence(orig_img, gen_img, mask, kernel_size=3):
    # Get edges
    orig_edges = cv2.Canny(orig_img, 100, 200)
    gen_edges = cv2.Canny(gen_img, 100, 200)

    # Create a border around the mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    mask_border = dilated_mask & ~mask

    # Count matching edges at the border
    if np.sum(mask_border) == 0:
        return 0

    orig_border_edges = orig_edges & mask_border
    gen_border_edges = gen_edges & mask_border

    # Count matching pixel positions
    matching_edges = np.sum((orig_border_edges > 0) & (gen_border_edges > 0))
    total_edges = np.sum(orig_border_edges > 0)

    return matching_edges / total_edges if total_edges > 0 else 0

def texture_consistency(orig_img, gen_img, mask, radius=3, n_points=24):
    if len(orig_img.shape) > 2:
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = orig_img
        gen_gray = gen_img

    # Compute LBP
    lbp_orig = local_binary_pattern(orig_gray, n_points, radius, method='uniform')
    lbp_gen = local_binary_pattern(gen_gray, n_points, radius, method='uniform')

    # Only compare the inpainted region
    lbp_orig_masked = lbp_orig[mask > 0]
    lbp_gen_masked = lbp_gen[mask > 0]

    if len(lbp_orig_masked) == 0:
        return 0

    # Calculate histograms
    n_bins = n_points + 2
    hist_orig, _ = np.histogram(lbp_orig_masked, bins=n_bins, range=(0, n_bins), density=True)
    hist_gen, _ = np.histogram(lbp_gen_masked, bins=n_bins, range=(0, n_bins), density=True)

    # Compare histograms using chi-square distance
    chi_square = 0.5 * np.sum(((hist_orig - hist_gen) ** 2) / (hist_orig + hist_gen + 1e-10))

    # Convert to a similarity score (0-1 range)
    return np.exp(-chi_square)

def calc_hist_similarity(img1, img2):
    hist1_b = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist1_g = cv2.calcHist([img1], [1], None, [256], [0, 256])
    hist1_r = cv2.calcHist([img1], [2], None, [256], [0, 256])

    hist2_b = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist2_g = cv2.calcHist([img2], [1], None, [256], [0, 256])
    hist2_r = cv2.calcHist([img2], [2], None, [256], [0, 256])

    cv2.normalize(hist1_b, hist1_b, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist1_g, hist1_g, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist1_r, hist1_r, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_b, hist2_b, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_g, hist2_g, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_r, hist2_r, 0, 1, cv2.NORM_MINMAX)

    score_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
    score_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
    score_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)

    return (score_b + score_g + score_r) / 3.0

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

        # Add pattern continuity
        pattern_score = pattern_continuity(input_image, output_image, binary_mask)
        metrics['pattern_continuity'] = round(pattern_score, 4)

        # Add edge coherence
        edge_score = edge_coherence(input_gray, output_gray, binary_mask)
        metrics['edge_coherence'] = round(edge_score, 4)

        # Add texture consistency
        texture_score = texture_consistency(input_image, output_image, binary_mask)
        metrics['texture_consistency'] = round(texture_score, 4)

        # In calculate_metrics(), after calculating global metrics:
        hist_similarity = calc_hist_similarity(input_image, output_image)
        metrics['histogram_similarity'] = round(hist_similarity, 4)

        # Calculate overall score
        overall_score = (
            0.3 * masked_ssim_value +
            0.1 * min(1.0, psnr_score/50) +
            0.2 * pattern_score +
            0.15 * edge_score +
            0.15 * texture_score +
            0.1 * (1.0 - min(1.0, masked_color_diff/30))
        )
        metrics['overall_score'] = round(overall_score, 4)

        # Add quality rating
        if overall_score > 0.85:
            rating = "Excellent"
        elif overall_score > 0.7:
            rating = "Good"
        elif overall_score > 0.5:
            rating = "Fair"
        else:
            rating = "Poor"
        metrics['quality_rating'] = rating

        # Calculate improvement percentage
        masked_input_mse = np.mean(((input_image_rgb[inpaint_mask] - masked_input_rgb[inpaint_mask])**2))
        if masked_input_mse > 0:
            improvement = (masked_input_mse - masked_mse) / masked_input_mse * 100
            metrics['improvement_percentage'] = improvement
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

# Create a message queue for progress updates
progress_queue = queue.Queue()

def send_progress(message):
    progress_queue.put(message)

# Update the progress route to handle completion properly
@app.route('/progress')
def progress():
    def generate():
        while True:
            try:
                message = progress_queue.get(timeout=60)  # Add timeout
                if message == "DONE":
                    break
                yield f"data: {message}\n\n"
            except queue.Empty:
                # If no message received for 60 seconds, end the stream
                break
    return Response(generate(), mimetype='text/event-stream')

# Update the upload route to ensure proper progress handling
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # --- Step 1: Handle input image upload ---
        if 'input_image' not in request.files:
            return "No input image provided", 400
        
        input_file = request.files['input_image']
        if input_file and allowed_file(input_file.filename):
            # Get original extension
            ext = os.path.splitext(input_file.filename)[1].lower()
            # Save as input_img with original extension
            input_filename = f'input_img{ext}'
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            input_file.save(input_path)
        else:
            return "Invalid input image", 400

        # --- Step 2: Handle mask selection ---
        mask_option = request.form.get('mask_option')
        
        if mask_option == 'upload_mask':
            mask_file = request.files.get('mask_image')
            if mask_file and allowed_file(mask_file.filename):
                # Save mask with original extension
                ext = os.path.splitext(mask_file.filename)[1].lower()
                mask_filename = f'mask{ext}'
                mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
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
        elif mask_option == 'pottery_cracks':
            intensity = float(request.form.get('intensity', 0.7))
            import edge_detection_mask
            edge_detection_mask.create_pottery_crack_mask(
                os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png'),
                intensity=intensity
            )
        elif mask_option == 'combined_edge':
            import edge_detection_mask
            edge_detection_mask.create_combined_edge_mask(
                os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
            )
        elif mask_option == 'outer_edge':
            import edge_detection_mask
            edge_detection_mask.create_outer_edge_mask(
                os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
            )
        elif mask_option == 'broken_edge':
            import edge_detection_mask
            edge_detection_mask.create_broken_outer_edge_mask(
                os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
            )
        elif mask_option == 'stylized_edge':
            import edge_detection_mask
            edge_detection_mask.create_stylized_pottery_edge_mask(
                os.path.join(app.config['UPLOAD_FOLDER'], 'input_img.png')
            )
        else:
            return "Invalid mask option", 400

        # After handling mask selection and before running inpainting
        # Create masked input image
        create_masked_input()

        # --- Step 3: Run the inpainting process ---
        def run_inpainting():
            try:
                send_progress("Creating generator...")
                subprocess.run(['python', 'inpaint.py'], check=True)
                send_progress("Inpainting complete!")
                send_progress("DONE")
            except Exception as e:
                send_progress(f"Error: {str(e)}")
                send_progress("DONE")

        thread = threading.Thread(target=run_inpainting)
        thread.daemon = True  # Make thread daemon so it doesn't block shutdown
        thread.start()
        
        return "", 200
    except Exception as e:
        return str(e), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def create_masked_input():
    # Get the input image filename (with extension)
    input_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                  if f.startswith('input_img')]
    if not input_files:
        return None
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_files[0])
    mask_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                 if f.startswith('mask')]
    if not mask_files:
        return None
    
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_files[0])
    
    # Create masked input image
    input_image = cv2.imread(input_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize both to 512x512
    input_image = cv2.resize(input_image, (512, 512))
    mask_image = cv2.resize(mask_image, (512, 512))
    
    # Create binary mask and apply it
    _, binary_mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
    masked_input = input_image.copy()
    masked_input[binary_mask == 255] = [255, 255, 255]
    
    # Save the masked input image in the upload folder
    masked_input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'masked_input.png')
    cv2.imwrite(masked_input_path, masked_input)
    
    return masked_input_path

@app.route('/masked_input')
def masked_input():
    masked_input_path = create_masked_input()
    if masked_input_path is None:
        return "Could not create masked input image", 404
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'masked_input.png')

@app.route('/results')
def results():
    # Calculate metrics before rendering the results page
    metrics = calculate_metrics()
    return render_template('results.html', metrics=metrics)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render-assigned port or default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)

