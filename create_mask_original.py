import numpy as np
import cv2
from config import RESIZE_TO, MASKIMAGE

# image mask

# free form mask
# bbox mask


def create_ff_mask():
    # config = {
    #     "img_shape": list(RESIZE_TO),
    #     "mv": 15,
    #     "ma": 4.0,
    #     "ml": 40,
    #     "mbw": 5,
    # }

    # h, w = config["img_shape"]
    # mask = np.zeros((h, w))
    # num_v = np.random.randint(config["mv"])

    # for i in range(num_v):
    #     start_x = np.random.randint(w)
    #     start_y = np.random.randint(h)
    #     for j in range(1 + np.random.randint(5)):
    #         angle = 0.01 + np.random.randint(config["ma"])
    #         if i % 2 == 0:
    #             angle = 2 * 3.1415926 - angle
    #         length = 10 + np.random.randint(config["ml"])
    #         brush_w = 5 + np.random.randint(config["mbw"])
    #         end_x = (start_x + length * np.sin(angle)).astype(np.int32)
    #         end_y = (start_y + length * np.cos(angle)).astype(np.int32)

    #         cv2.line(mask, (start_y, start_x), (end_y, end_x), 255.0, brush_w)
    #         start_x, start_y = end_x, end_y
    config = {
        "img_shape": list(RESIZE_TO),
        "num_patches": 1,  # Number of circular patches
        "min_radius": 100,  # Minimum radius of the circle
        "max_radius": 250,  # Maximum radius of the circle
        "x_range": (100, 400),  # Random X coordinate range for the center
        "y_range": (100, 400),  # Random Y coordinate range for the center
    }

    h, w = config["img_shape"]
    mask = np.zeros((h, w), dtype=np.uint8)  # Initialize empty mask
    
    for i in range(config["num_patches"]):
        # Random position for the center of the circle within the specified range
        center_x = np.random.randint(config["x_range"][0], config["x_range"][1])
        center_y = np.random.randint(config["y_range"][0], config["y_range"][1])

        # Random radius size for the circle
        radius = np.random.randint(config["min_radius"], config["max_radius"])

        # Draw a filled circle (white patch) on the mask
        cv2.circle(mask, (100, 250), 50, 255, -1)  # -1 for filled circle

    mask = mask.astype(np.uint8)
    cv2.imwrite(MASKIMAGE, mask)


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
