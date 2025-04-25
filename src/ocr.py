# version: 0.3.2
# path: src/ocr.py

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import threading
import queue

class OcrEngine:
    def __init__(self, use_angle_cls=True, lang='en'):
        """
        Initialize the PaddleOCR engine.
        :param use_angle_cls: Enable angle classification.
        :param lang: Language for OCR model, e.g., 'en' or 'ch'.
        """
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def extract_text(self, img):
        """
        Extract text from an image using PaddleOCR.
        :param img: Input image as a numpy array or PIL Image.
        :return: Concatenated text string extracted from the image.
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        result = self.ocr.ocr(img, cls=True)
        lines = []
        for line in result[0]:
            text, confidence = line[1][0], line[1][1]
            if confidence > 0.5:
                lines.append(text)
        return '\n'.join(lines)

    def extract_text_with_boxes(self, img):
        """
        Extract text along with bounding boxes.
        :param img: Input image as a numpy array or PIL Image.
        :return: List of dicts: [{'text': str, 'confidence': float, 'box': list}]
        """
        if isinstance(img, Image.Image):
            img = np.array(img)

        result = self.ocr.ocr(img, cls=True)
        output = []
        for line in result[0]:
            box, (text, confidence) = line[0], line[1]
            if confidence > 0.5:
                output.append({'text': text, 'confidence': confidence, 'box': box})
        return output

    def extract_text_batch(self, images):
        """
        Extract text from a batch of images using PaddleOCR.
        :param images: List of images (numpy arrays or PIL Images).
        :return: List of text results for each image.
        """
        results = []
        for img in images:
            results.append(self.extract_text(img))
        return results

    def extract_text_batch_threaded(self, images):
        """
        Extract text from a batch of images using multi-threading.
        :param images: List of images (numpy arrays or PIL Images).
        :return: List of text results for each image.
        """
        results = [None] * len(images)

        def worker(idx, img):
            results[idx] = self.extract_text(img)

        threads = []
        for idx, img in enumerate(images):
            thread = threading.Thread(target=worker, args=(idx, img))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def queue_image_for_processing(self, img):
        """
        Add an image to the processing queue.
        :param img: Image to process.
        """
        self.task_queue.put(img)

    def get_processed_result(self):
        """
        Retrieve the next processed result from the result queue.
        :return: OCR result or None if no result is available.
        """
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def _process_queue(self):
        """
        Internal method to process the OCR task queue in a background thread.
        """
        while True:
            img = self.task_queue.get()
            result = self.extract_text(img)
            self.result_queue.put(result)
            self.task_queue.task_done()
