# version: 0.3.7
# path: src/ocr.py

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - allow dummy PIL stubs
    Image = None
import threading
import queue

# Attempt to import PaddleOCR, handle all import errors
try:
    from paddleocr import PaddleOCR

    _has_paddle = True
except Exception:
    _has_paddle = False


class OcrEngine:
    def __init__(self, use_angle_cls=True, lang="en"):
        """Initialize the PaddleOCR engine."""
        if _has_paddle:
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        else:  # pragma: no cover - fallback for tests without PaddleOCR
            self.ocr = None
        # Setup threading queues
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def extract_text(self, img):
        """Extract text from an image."""
        if self.ocr is None:
            return ""
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img
        result = self.ocr.ocr(arr, cls=True)
        lines = []
        for line in result[0]:
            text, conf = line[1]
            if conf > 0.5:
                lines.append(text)
        return "\n".join(lines)

    def extract_data(self, img, conf_threshold=0.5):
        """Return OCR results with bounding boxes similar to pytesseract."""
        if self.ocr is None:
            return []
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img
        result = self.ocr.ocr(arr, cls=True)
        boxes = []
        for line in result[0]:
            box, (text, conf) = line
            if conf < conf_threshold:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            left, top = min(xs), min(ys)
            width, height = max(xs) - left, max(ys) - top
            boxes.append(
                {
                    "text": text,
                    "left": int(left),
                    "top": int(top),
                    "width": int(width),
                    "height": int(height),
                }
            )
        return boxes

    def extract_text_batch(self, images):
        """
        Sequential batch OCR.
        """
        return [self.extract_text(img) for img in images]

    def extract_text_batch_threaded(self, images):
        """
        Threaded batch OCR.
        """
        results = [None] * len(images)

        def worker(idx, image):
            results[idx] = self.extract_text(image)

        threads = []
        for idx, image in enumerate(images):
            t = threading.Thread(target=worker, args=(idx, image))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    def queue_image_for_processing(self, img):
        """
        Queue image for async OCR.
        """
        self.task_queue.put(img)

    def get_processed_result(self):
        """
        Retrieve next async OCR result.
        """
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def _process_queue(self):
        while True:
            img = self.task_queue.get()
            text = self.extract_text(img)
            self.result_queue.put(text)
            self.task_queue.task_done()
