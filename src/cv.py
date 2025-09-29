# version: 0.4.0
# path: src/cv.py

import logging
import threading
import queue
import difflib
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - allow dummy PIL stubs
    Image = None

from .ocr import OcrEngine
from .detector import (
    YoloOnnxDetector,
    load_detector_settings,
)


LOGGER = logging.getLogger(__name__)


class CvEngine:
    def __init__(self, ocr=None, detector: Optional[YoloOnnxDetector] = None):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.lock_template = cv2.imread("templates/is_target_locked.png", 0)
        self.lock_threshold = 0.8
        self.ocr = ocr or OcrEngine()
        self.detector_cfg = load_detector_settings()
        self.detector = detector or self._build_detector(self.detector_cfg)

    def _build_detector(self, cfg: Dict) -> Optional[YoloOnnxDetector]:
        model_path = cfg.get("model_path") if isinstance(cfg, dict) else None
        class_map = cfg.get("class_names") if isinstance(cfg, dict) else None
        if not model_path or not class_map:
            return None
        providers_raw = cfg.get("providers")
        providers = providers_raw if isinstance(providers_raw, list) else None
        try:
            numeric_map = {int(k): str(v) for k, v in class_map.items()}
            return YoloOnnxDetector(model_path, numeric_map, providers=providers)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Falling back to template detection: %s", exc)
            return None

    def detect_elements(
        self,
        img,
        templates: Optional[Dict[str, Union[str, Dict[str, object]]]],
        threshold: float = 0.25,
        multi_scale: bool = False,
        scales: Optional[Iterable[float]] = None,
    ) -> List[Dict[str, object]]:
        """Detect UI elements using the configured YOLO pipeline.

        ``templates`` accepts a mapping of ROI identifiers to configuration
        dictionaries. Each configuration may include ``roi`` coordinates and a
        ``labels`` iterable. When a string is provided the call falls back to
        legacy template matching for that entry.
        """

        if not templates:
            return []

        if isinstance(img, Image.Image):
            img = np.array(img)

        if self.detector is None:
            return self._detect_with_templates(
                img,
                templates,
                threshold=threshold,
                multi_scale=multi_scale,
                scales=scales,
            )

        results: List[Dict[str, object]] = []
        for roi_name, target in templates.items():
            if isinstance(target, str):
                results.extend(
                    self._detect_with_templates(
                        img,
                        {roi_name: target},
                        threshold=threshold,
                        multi_scale=multi_scale,
                        scales=scales,
                    )
                )
                continue

            target_cfg = target or {}
            roi = target_cfg.get("roi") if isinstance(target_cfg, dict) else None
            labels = target_cfg.get("labels") if isinstance(target_cfg, dict) else None
            roi_box = None
            if isinstance(roi, (tuple, list)) and len(roi) == 4:
                roi_box = tuple(int(v) for v in roi)
            crop = (
                img
                if roi_box is None
                else img[roi_box[1] : roi_box[3], roi_box[0] : roi_box[2]]
            )
            detections = self.detector.predict(crop, conf_threshold=threshold)
            for det in detections:
                if labels and det.label not in labels:
                    continue
                bbox = det.bbox
                if roi_box is not None:
                    bbox = (
                        bbox[0] + roi_box[0],
                        bbox[1] + roi_box[1],
                        bbox[2] + roi_box[0],
                        bbox[3] + roi_box[1],
                    )
                results.append(
                    {
                        "name": det.label,
                        "roi": roi_name,
                        "bbox": bbox,
                        "confidence": det.confidence,
                        "class_id": det.class_id,
                    }
                )
        return results

    def _detect_with_templates(
        self,
        img,
        templates: Dict[str, Union[str, Dict[str, object]]],
        threshold: float = 0.8,
        multi_scale: bool = False,
        scales: Optional[Iterable[float]] = None,
    ) -> List[Dict[str, object]]:
        if isinstance(img, Image.Image):
            img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected: List[Dict[str, object]] = []
        for name, template_spec in templates.items():
            template_path = (
                template_spec
                if isinstance(template_spec, str)
                else template_spec.get("template")
                if isinstance(template_spec, dict)
                else None
            )
            if not template_path:
                continue
            template_orig = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_orig is None:
                continue
            scale_list = list(scales) if (multi_scale and scales) else [1.0]
            for scale in scale_list:
                template = cv2.resize(
                    template_orig,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < threshold:
                    continue
                x, y = max_loc
                h, w = template.shape
                detected.append(
                    {
                        "name": name,
                        "roi": name,
                        "bbox": (x, y, x + w, y + h),
                        "confidence": float(max_val),
                        "class_id": -1,
                    }
                )
        return detected

    def is_module_active(self, slot_img):
        """
        Detect whether a ship module (laser, etc.) is currently active based on brightness.
        Active modules generally glow (higher intensity).
        """
        gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return brightness > 75  # You can tune this based on actual visuals

    def find_asteroid_entry(self, img, ore_names=None):
        """Locate an asteroid row inside an overview panel using OCR.

        Parameters
        ----------
        img : ndarray or PIL.Image
            Cropped screenshot of the overview panel.
        ore_names : list, optional
            List of ore names to match against. If ``None`` a default list of
            common ores is used.

        Returns
        -------
        tuple or None
            Coordinates ``(x, y)`` relative to ``img`` of the matched word
            centre or ``None`` if no ore text was found.
        """
        default_ores = [
            "veldspar",
            "scordite",
            "plagioclase",
            "pyroxeres",
            "omber",
            "kernite",
            "jaspet",
            "hemorphite",
            "hedbergite",
            "gneiss",
            "ochre",
            "spodumain",
            "mercoxit",
        ]
        ore_names = [o.lower() for o in (ore_names or default_ores)]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = self.ocr.extract_data(gray)

        for entry in data:
            word = entry["text"].strip().lower()
            if not word:
                continue
            if word in ore_names or difflib.get_close_matches(
                word, ore_names, n=1, cutoff=0.7
            ):
                x, y, w, h = (
                    entry["left"],
                    entry["top"],
                    entry["width"],
                    entry["height"],
                )
                return (x + w // 2, y + h // 2)
        return None

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
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

    def highlight_elements(self, img, elements, template_shapes=None):
        """
        Draw rectangles around detected elements for visualization.
        :param img: Input image as a numpy array or PIL Image.
        :param elements: List of detected elements.
        :param template_shapes: Optional dict of template shapes {'name': (w, h)}.
        :return: Annotated image.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        annotated_img = img.copy()
        for elem in elements:
            if "bbox" in elem:
                x1, y1, x2, y2 = map(int, elem["bbox"])
            elif template_shapes and elem.get("name") in template_shapes:
                x, y = elem.get("position", (0, 0))
                w, h = template_shapes[elem["name"]]
                scale = elem.get("scale", 1.0)
                w, h = int(w * scale), int(h * scale)
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                continue
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return annotated_img

    def detect_target_lock(self, img):
        """
        Match the red crosshair target-lock icon inside the is_target_locked ROI.
        Returns True if locked target is detected.
        """
        if img is None or self.lock_template is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, self.lock_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > self.lock_threshold

    def queue_image_for_detection(
        self, img, templates, threshold=0.8, multi_scale=False, scales=None
    ):
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
            result = self.detect_elements(
                img, templates, threshold, multi_scale, scales
            )
            self.result_queue.put(result)
            self.task_queue.task_done()
