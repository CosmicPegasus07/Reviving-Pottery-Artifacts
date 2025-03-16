import numpy as np
import cv2
from config import RESIZE_TO, MASKIMAGE


def create_outer_edge_mask(input_image_path, thickness=5, simplify_factor=0.02):
    """
    Creates a mask focusing only on the outer edges/boundary of pottery,
    excluding internal details.
    
    Args:
        input_image_path (str): Path to the input image
        thickness (int): Thickness of the edge mask
        simplify_factor (float): Factor to simplify contours (0.0-1.0)
                                 Higher values create simpler contours
        
    Returns:
        numpy.ndarray: Binary mask image with only outer edges
    """
    # Read and resize the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_image_path}")
    
    image = cv2.resize(image, (RESIZE_TO[1], RESIZE_TO[0]))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to binarize the image
    # Adjust these threshold values based on your images
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Optional: Morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Invert if needed (depends on your image - pottery should be white)
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros((RESIZE_TO[0], RESIZE_TO[1]), dtype=np.uint8)
    
    # Sort contours by area and keep only the largest ones (likely the pottery)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # If we have contours, process the largest one(s)
    if contours:
        # Take only the largest contour or a few largest
        largest_contours = contours[:1]  # Just take the largest one
        
        for contour in largest_contours:
            # Simplify the contour to get a cleaner outline
            epsilon = simplify_factor * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw just the contour with specified thickness
            cv2.drawContours(mask, [approx_contour], -1, 255, thickness)
    
    # Save the final mask
    cv2.imwrite(MASKIMAGE, mask)
    
    return mask

def create_broken_outer_edge_mask(input_image_path, thickness=5, 
                                  simplify_factor=0.02, gap_probability=0.3):
    """
    Creates a mask with the outer edges/boundary of pottery that includes
    random breaks or gaps to simulate cracks for broken pottery.
    
    Args:
        input_image_path (str): Path to the input image
        thickness (int): Thickness of the edge mask
        simplify_factor (float): Factor to simplify contours (0.0-1.0)
        gap_probability (float): Probability of creating gaps (0.0-1.0)
        
    Returns:
        numpy.ndarray: Binary mask image with broken outer edges
    """
    # Read and resize the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_image_path}")
    
    image = cv2.resize(image, (RESIZE_TO[1], RESIZE_TO[0]))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to binarize the image
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Optional: Morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Invert if needed (depends on your image - pottery should be white)
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros((RESIZE_TO[0], RESIZE_TO[1]), dtype=np.uint8)
    
    # Sort contours by area and keep only the largest ones (likely the pottery)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # If we have contours, process the largest one(s)
    if contours:
        # Take only the largest contour or a few largest
        largest_contours = contours[:1]  # Just take the largest one
        
        for contour in largest_contours:
            # Simplify the contour to get a cleaner outline
            epsilon = simplify_factor * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert the contour to a list of points
            points = approx_contour.reshape(-1, 2).tolist()
            
            # Create "broken" edges by drawing line segments with gaps
            for i in range(len(points)):
                # Get current point and next point (wrap around to first point for last edge)
                current_point = tuple(points[i])
                next_point = tuple(points[(i + 1) % len(points)])
                
                # Randomly decide whether to draw this segment
                if np.random.random() > gap_probability:
                    cv2.line(mask, current_point, next_point, 255, thickness)
            
            # Add a few random "broken pieces" - small line segments near the contour
            num_breaks = int(len(points) * 0.3)  # About 30% of the number of points
            for _ in range(num_breaks):
                # Select a random point from the contour
                idx = np.random.randint(0, len(points))
                point = points[idx]
                
                # Create a random offset
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.randint(10, 50)
                
                # Calculate end point of the break
                end_x = int(point[0] + distance * np.cos(angle))
                end_y = int(point[1] + distance * np.sin(angle))
                
                # Ensure endpoints are within image bounds
                end_x = max(0, min(end_x, RESIZE_TO[1] - 1))
                end_y = max(0, min(end_y, RESIZE_TO[0] - 1))
                
                # Draw the break line
                cv2.line(mask, tuple(point), (end_x, end_y), 255, 
                         np.random.randint(1, thickness + 1))
    
    # Optional: Add a few random impact points
    if np.random.random() < 0.5:  # 50% chance to add impact points
        num_impacts = np.random.randint(1, 3)
        
        for _ in range(num_impacts):
            # Random position near the contour
            if contours and len(contours[0]) > 0:
                # Choose a random point from the largest contour
                idx = np.random.randint(0, len(contours[0]))
                point = contours[0][idx][0]
                
                # Add random offset
                offset_x = np.random.randint(-30, 30)
                offset_y = np.random.randint(-30, 30)
                
                center_x = point[0] + offset_x
                center_y = point[1] + offset_y
                
                # Ensure center is within image bounds
                center_x = max(0, min(center_x, RESIZE_TO[1] - 1))
                center_y = max(0, min(center_y, RESIZE_TO[0] - 1))
                
                # Random radius
                radius = np.random.randint(5, 15)
                
                # Draw the impact point
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Save the final mask
    cv2.imwrite(MASKIMAGE, mask)
    
    return mask

def create_stylized_pottery_edge_mask(input_image_path, stylization_level=5, thickness=5):
    """
    Creates a mask that focuses on stylized outer edges of pottery with artistic
    simplification and potential for randomized breaks.
    
    Args:
        input_image_path (str): Path to the input image
        stylization_level (int): Level of stylization (1-10)
                                  Higher values create more simplified/artistic outlines
        thickness (int): Thickness of the edge mask
        
    Returns:
        numpy.ndarray: Binary mask image with stylized pottery edges
    """
    # Read and resize the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_image_path}")
    
    image = cv2.resize(image, (RESIZE_TO[1], RESIZE_TO[0]))
    
    # Apply bilateral filtering for edge-preserving smoothing
    # Adjust these parameters for more or less smoothing
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply stylization (edges with artistic look)
    if stylization_level > 0:
        # Convert stylization level to appropriate parameters
        sigma_s = 30 + stylization_level * 10  # Spatial sigma
        sigma_r = 0.15 + stylization_level * 0.02  # Range sigma
        
        # Edge-preserving filter to create a more stylized look
        filtered = cv2.edgePreservingFilter(
            filtered, 
            flags=cv2.RECURS_FILTER, 
            sigma_s=sigma_s, 
            sigma_r=sigma_r
        )
    
    # Convert to grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to binarize the image with Otsu's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Optional: Apply morphological operations to clean the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Invert if needed (depends on your image - pottery should be white)
    if np.sum(binary == 0) > np.sum(binary == 255):
        binary = cv2.bitwise_not(binary)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros((RESIZE_TO[0], RESIZE_TO[1]), dtype=np.uint8)
    
    # Sort contours by area and keep only the largest ones (likely the pottery)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # If we have contours, process the largest one(s)
    if contours:
        # Take only a few largest contours - just the biggest if we only want the outer edge
        num_contours = 1  # Adjust if you want more than just the largest contour
        for i, contour in enumerate(contours[:num_contours]):
            # Simplify the contour based on stylization level
            # Higher stylization = more simplification
            epsilon = (0.005 + 0.005 * stylization_level) * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw the outline with specified thickness
            cv2.drawContours(mask, [approx_contour], -1, 255, thickness)
            
            # Optionally add some artistic breaks or variations to the contour
            if stylization_level > 5:
                # Convert the contour to a list of points
                points = approx_contour.reshape(-1, 2).tolist()
                
                # For high stylization, add some deliberate breaks/gaps
                for i in range(len(points)):
                    if np.random.random() < 0.2:  # 20% chance of gap
                        continue  # Skip this segment
                    
                    # Get current point and next point
                    current_point = tuple(points[i])
                    next_point = tuple(points[(i + 1) % len(points)])
                    
                    # Draw line with varying thickness
                    varying_thickness = max(1, thickness + np.random.randint(-2, 3))
                    cv2.line(mask, current_point, next_point, 255, varying_thickness)
    
    # Save the final mask
    cv2.imwrite(MASKIMAGE, mask)
    
    return mask

def create_edge_mask(input_image_path, threshold1=50, threshold2=150, 
                     min_edge_length=50, edge_thickness=3, 
                     mask_width=10, random_seed=None):
    """
    Create a mask based on edge detection from an input image.
    
    Args:
        input_image_path (str): Path to the input image
        threshold1 (int): First threshold for Canny edge detector
        threshold2 (int): Second threshold for Canny edge detector
        min_edge_length (int): Minimum length of edges to keep
        edge_thickness (int): Thickness of detected edges
        mask_width (int): Width of the mask around detected edges
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Binary mask image
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_image_path}")
    
    # Resize image if needed
    image = cv2.resize(image, (RESIZE_TO[1], RESIZE_TO[0]))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Dilate edges to make them thicker
    kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    mask = np.zeros((RESIZE_TO[0], RESIZE_TO[1]), dtype=np.uint8)
    
    # Filter contours by length and draw selected ones with thickness
    for contour in contours:
        # Calculate contour length
        contour_length = cv2.arcLength(contour, closed=True)
        
        # Only keep longer edges
        if contour_length > min_edge_length:
            # Select random segments of contours to mask (for broken pottery effect)
            if np.random.random() > 0.3:  # 70% chance to keep a qualifying contour
                cv2.drawContours(mask, [contour], -1, 255, mask_width)
    
    # Add some randomness to the mask for more natural broken edges
    height, width = mask.shape
    num_random_cracks = np.random.randint(1, 4)
    
    for _ in range(num_random_cracks):
        # Random starting point
        start_x = np.random.randint(width // 4, width * 3 // 4)
        start_y = np.random.randint(height // 4, height * 3 // 4)
        
        # Random ending point
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(50, 150)
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))
        
        # Ensure endpoints are within image bounds
        end_x = max(0, min(end_x, width - 1))
        end_y = max(0, min(end_y, height - 1))
        
        # Draw the random crack
        cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, mask_width)
    
    # Save the mask
    cv2.imwrite(MASKIMAGE, mask)
    
    return mask

def create_combined_edge_mask(input_image_path, circle_probability=0.3, edge_thickness=5):
    """
    Creates a mask that combines edge detection with occasional circular masks.
    
    Args:
        input_image_path (str): Path to the input image
        circle_probability (float): Probability of adding circular masks
        edge_thickness (int): Thickness of the edge masks
        
    Returns:
        numpy.ndarray: Binary mask image
    """
    # Create the edge-based mask
    mask = create_edge_mask(input_image_path, 
                           threshold1=30, 
                           threshold2=100,
                           min_edge_length=40,
                           edge_thickness=edge_thickness,
                           mask_width=8)
    
    # Potentially add some circular masks to simulate impact points
    h, w = mask.shape
    
    # Decide if we'll add circular masks
    if np.random.random() < circle_probability:
        num_circles = np.random.randint(1, 3)
        
        for _ in range(num_circles):
            # Random position for circle center
            center_x = np.random.randint(w // 4, w * 3 // 4)
            center_y = np.random.randint(h // 4, h * 3 // 4)
            
            # Random radius
            radius = np.random.randint(20, 60)
            
            # Draw the circle on the mask
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Add random cracks emanating from the circle
            num_cracks = np.random.randint(2, 5)
            for i in range(num_cracks):
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(radius, radius * 3)
                end_x = int(center_x + length * np.cos(angle))
                end_y = int(center_y + length * np.sin(angle))
                
                # Ensure endpoints are within image bounds
                end_x = max(0, min(end_x, w - 1))
                end_y = max(0, min(end_y, h - 1))
                
                # Draw the crack
                cv2.line(mask, (center_x, center_y), (end_x, end_y), 255, 
                         np.random.randint(2, edge_thickness + 1))
    
    # Save the final mask
    cv2.imwrite(MASKIMAGE, mask)
    
    return mask

# Example usage with advanced edge and texture detection for better pottery cracks
def create_pottery_crack_mask(input_image_path, intensity=0.7):
    """
    Creates a mask specifically designed for pottery cracks using multiple edge detection techniques.
    
    Args:
        input_image_path (str): Path to the input image
        intensity (float): Intensity of the crack effect (0.0 to 1.0)
        
    Returns:
        numpy.ndarray: Binary mask image for pottery cracks
    """
    # Read and resize the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_image_path}")
    
    image = cv2.resize(image, (RESIZE_TO[1], RESIZE_TO[0]))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple edge detection techniques for more accurate crack detection
    
    # 1. Canny edge detection
    edges_canny = cv2.Canny(gray, 30, 100)
    
    # 2. Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, sobel_thresh = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)
    
    # 3. Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, laplacian_thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Combine different edge detections with different weights
    combined_edges = cv2.addWeighted(edges_canny, 0.4, sobel_thresh, 0.3, 0)
    combined_edges = cv2.addWeighted(combined_edges, 0.7, laplacian_thresh, 0.3, 0)
    
    # Create a mask based on the combined edges
    mask = np.zeros_like(combined_edges)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Filter and draw the contours that look like cracks
    for contour in contours:
        # Calculate contour properties
        contour_length = cv2.arcLength(contour, closed=False)
        area = cv2.contourArea(contour)
        
        # Skip very small contours
        if contour_length < 20:
            continue
        
        # Skip contours that are too circular (not crack-like)
        if area > 0 and contour_length > 0:
            circularity = 4 * np.pi * area / (contour_length * contour_length)
            if circularity > 0.5:  # More likely a circle than a crack
                continue
        
        # Add randomness based on intensity
        if np.random.random() < intensity:
            # Draw the contour with varying thickness
            thickness = np.random.randint(2, 5)
            cv2.drawContours(mask, [contour], -1, 255, thickness)
    
    # Add some simulated impact points and connecting cracks
    h, w = mask.shape
    if np.random.random() < 0.7:  # 70% chance to add impact points
        num_impacts = np.random.randint(1, 3)
        
        for _ in range(num_impacts):
            # Random position for impact center
            center_x = np.random.randint(w // 4, w * 3 // 4)
            center_y = np.random.randint(h // 4, h * 3 // 4)
            
            # Random radius for impact point
            radius = np.random.randint(10, 40)
            
            # Draw the impact point
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Add cracks emanating from the impact point
            num_cracks = np.random.randint(3, 8)
            for i in range(num_cracks):
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(radius * 2, radius * 6)
                
                # Add some curvature to the cracks
                points = []
                points.append((center_x, center_y))
                
                current_x, current_y = center_x, center_y
                current_angle = angle
                
                # Create a curved crack with multiple segments
                for j in range(np.random.randint(3, 8)):
                    segment_length = length / (j + 1)
                    # Add some randomness to the angle
                    current_angle += np.random.uniform(-0.3, 0.3)
                    
                    next_x = int(current_x + segment_length * np.cos(current_angle))
                    next_y = int(current_y + segment_length * np.sin(current_angle))
                    
                    # Ensure endpoints are within image bounds
                    next_x = max(0, min(next_x, w - 1))
                    next_y = max(0, min(next_y, h - 1))
                    
                    points.append((next_x, next_y))
                    current_x, current_y = next_x, next_y
                
                # Draw the multi-segment crack
                for j in range(len(points) - 1):
                    thickness = np.random.randint(1, 4)
                    cv2.line(mask, points[j], points[j+1], 255, thickness)
    
    # Save the final mask
    cv2.imwrite(MASKIMAGE, mask)
    
    return mask


