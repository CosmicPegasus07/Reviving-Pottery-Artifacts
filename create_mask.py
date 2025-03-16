import numpy as np
import cv2
from config import RESIZE_TO, MASKIMAGE
from ipywidgets import widgets, Layout, Button
from IPython.display import display, clear_output

def create_ff_mask():
    """Create a random free-form mask with multiple strokes"""
    config = {
        "img_shape": list(RESIZE_TO),
        "mv": 15,  # max number of vertices
        "ma": 4.0,  # max angle
        "ml": 40,  # max length
        "mbw": 5,  # max brush width
    }
    
    h, w = config["img_shape"]
    mask = np.zeros((h, w), dtype=np.uint8)
    num_v = np.random.randint(1, config["mv"])
    
    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.random() * config["ma"]
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(config["ml"])
            brush_w = 5 + np.random.randint(config["mbw"])
            end_x = int(start_x + length * np.sin(angle))
            end_y = int(start_y + length * np.cos(angle))
            
            # Ensure points are within image boundaries
            end_x = max(0, min(end_x, w-1))
            end_y = max(0, min(end_y, h-1))
            
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 255, brush_w)
            start_x, start_y = end_x, end_y

    cv2.imwrite(MASKIMAGE, mask)
    return mask

def create_circular_mask(center_x=None, center_y=None, radius=50):
    """Create a circular mask with specified center and radius"""
    h, w = RESIZE_TO
    
    # Use default center if not provided
    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
        
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # -1 for filled circle
    
    cv2.imwrite(MASKIMAGE, mask)
    return mask

def create_circular_mask_with_ui():
    """Create a circular mask with interactive UI controls for position and size"""
    h, w = RESIZE_TO
    output = widgets.Output()
    
    # Create sliders for controlling the circle parameters
    x_slider = widgets.IntSlider(
        value=w//2,
        min=0,
        max=w,
        step=1,
        description='X Position:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='300px')
    )
    
    y_slider = widgets.IntSlider(
        value=h//2,
        min=0,
        max=h,
        step=1,
        description='Y Position:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='300px')
    )
    
    radius_slider = widgets.IntSlider(
        value=50,
        min=5,
        max=min(h, w)//2,
        step=1,
        description='Radius:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=Layout(width='300px')
    )
    
    # Create a preview button and apply button
    preview_button = Button(description="Preview")
    apply_button = Button(description="Apply")
    
    # Display the UI components
    display(widgets.VBox([
        widgets.Label(value="Adjust Circle Parameters:"),
        x_slider,
        y_slider,
        radius_slider,
        widgets.HBox([preview_button, apply_button]),
        output
    ]))
    
    # Define button click handlers
    def on_preview_clicked(b):
        with output:
            clear_output()
            # Create a temporary preview image
            center_x = x_slider.value
            center_y = y_slider.value
            radius = radius_slider.value
            
            preview_img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.circle(preview_img, (center_x, center_y), radius, (255, 255, 255), -1)
            
            # Display the preview image
            from IPython.display import Image
            import io
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 8))
            plt.imshow(preview_img, cmap='gray')
            plt.title(f"Preview: Circle at ({center_x}, {center_y}) with radius {radius}")
            plt.axis('off')
            plt.show()
    
    def on_apply_clicked(b):
        with output:
            clear_output()
            # Create and save the mask
            center_x = x_slider.value
            center_y = y_slider.value
            radius = radius_slider.value
            
            mask = create_circular_mask(center_x, center_y, radius)
            print(f"Circular mask created at position ({center_x}, {center_y}) with radius {radius}")
            print("Mask saved in input folder as mask.png")
    
    # Connect the button click events
    preview_button.on_click(on_preview_clicked)
    apply_button.on_click(on_apply_clicked)

def create_bbox_mask():
    shape = list(RESIZE_TO)
    margin = [10, 10]
    bbox_shape = [30, 30]
    
    def random_bbox(shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                   VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height, img_width = shape
        height, width = bbox_shape
        ver_margin, hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)
    
    bboxs = []
    for i in range(20):
        bbox = random_bbox(shape, margin, bbox_shape)
        bboxs.append(bbox)
    
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    # print(mask.shape)
    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        mask[
            (bbox[0] + h) : (bbox[0] + bbox[2] - h),
            (bbox[1] + w) : (bbox[1] + bbox[3] - w),
        ] = 255.0
    
    mask = mask.astype(np.uint8)
    cv2.imwrite(MASKIMAGE, mask)