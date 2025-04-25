# version: 0.1.0
# path: src/capture_utils.py

import numpy as np
from PIL import ImageGrab

def capture_screen(region=None):
    """
    Capture a screenshot of the defined region or full screen.
    :param region: Optional screen region (x, y, width, height).
    :return: Screenshot as numpy array.
    """
    img = ImageGrab.grab(bbox=region)
    return np.array(img)
