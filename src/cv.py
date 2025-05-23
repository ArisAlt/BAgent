# version: 0.3.2
# path: src/cv.py

import cv2
import numpy as np
from PIL import Image
import threading
import queue

class CvEngine:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def detect_elements(self, img, templates, threshold=0.8, multi_scale=False, scales=None):
        """
        Detect UI elements in an image using template matching.
        :param img: Input image as a numpy array or PIL Image.
        :param templates: Dict of template names and file paths.
        :param threshold: Confidence threshold for detection.
        :param multi_scale: Enable multi-scale template matching.
        :param scales: List of scales to use for multi-scale matching.
        :return: List of detected elements [{'name': str, 'position': (x, y), 'confidence': float, 'scale': float}]
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_elements = []
        for name, template_path in templates.items():
            template_orig = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            scale_list = scales if multi_scale else [1.0]
            for scale in scale_list:
                template = cv2.resize(template_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > threshold:
                    detected_elements.append({
                        'name': name,
                        'position': max_loc,
                        'confidence': max_val,
                        'scale': scale
                    })
        return detected_elements

    def detect_contours(self, img, threshold1=100, threshold2=200):
        """
        Detect contours in the image for shape-based element detection.
        :param img: Input image as a numpy array or PIL Image.
        :param threshold1: First threshold for the hysteresis procedure in Canny.
        :param threshold2: Second threshold for the hysteresis procedure in Canny.
        :return: List of contours.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, threshold1, threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_by_color(self, img, lower_bound, upper_bound):
        """
        Filter image regions by color range.
        :param img: Input image as a numpy array or PIL Image.
        :param lower_bound: Lower bound for color in HSV.
        :param upper_bound: Upper bound for color in HSV.
        :return: Masked image regions.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
        return mask

    def highlight_elements(self, img, elements, template_shapes):
        """
        Draw rectangles around detected elements for visualization.
        :param img: Input image as a numpy array or PIL Image.
        :param elements: List of detected elements.
        :param template_shapes: Dict of template shapes {'name': (w, h)}.
        :return: Annotated image.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        annotated_img = img.copy()
        for elem in elements:
            x, y = elem['position']
            w, h = template_shapes[elem['name']]
            w, h = int(w * elem.get('scale', 1.0)), int(h * elem.get('scale', 1.0))
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return annotated_img

    def queue_image_for_detection(self, img, templates, threshold=0.8, multi_scale=False, scales=None):
        """
        Add an image and templates to the processing queue.
        :param img: Image to process.
        :param templates: Templates for detection.
        """
        self.task_queue.put((img, templates, threshold, multi_scale, scales))

    def get_detected_results(self):
        """
        Retrieve the next detection result from the result queue.
        :return: Detection result or None if no result is available.
        """
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def _process_queue(self):
        """
        Internal method to process the detection task queue in a background thread.
        """
        while True:
            img, templates, threshold, multi_scale, scales = self.task_queue.get()
            result = self.detect_elements(img, templates, threshold, multi_scale, scales)
            self.result_queue.put(result)
            self.task_queue.task_done()
