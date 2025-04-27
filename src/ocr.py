# version: 0.3.4
# path: src/ocr.py

import numpy as np
from PIL import Image, ImageGrab
import threading
import queue

# Attempt to import PaddleOCR, handle all import errors
try:
    from paddleocr import PaddleOCR
    _has_paddle = True
except Exception:
    _has_paddle = False

class OcrEngine:
    def __init__(self, use_angle_cls=True, lang='en', use_paddle=True):
        """
        Initialize OCR engine: try PaddleOCR if available and requested, else use pytesseract.
        """
        self.use_paddle = use_paddle and _has_paddle
        if self.use_paddle:
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        # Dynamically import pytesseract to avoid pandas import at top-level
        try:
            import pytesseract
            self.pytesseract = pytesseract
            # configure tesseract executable path as needed
            self.pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
            self.has_tesseract = True
        except Exception:
            self.has_tesseract = False
        # Setup threading queues
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def extract_text(self, img):
        """
        Extract text from image using selected OCR engine.
        """
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        # Use PaddleOCR if available
        if self.use_paddle:
            arr = np.array(pil_img)
            result = self.ocr.ocr(arr, cls=True)
            lines = []
            for line in result[0]:
                text, conf = line[1]
                if conf > 0.5:
                    lines.append(text)
            return '\n'.join(lines)
        # Fallback to pytesseract if available
        if self.has_tesseract:
            return self.pytesseract.image_to_string(pil_img)
        # No OCR available
        return ""

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
