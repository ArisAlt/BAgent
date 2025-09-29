# version: 0.1.0
# path: src/detector.py
"""ONNX-backed object detection utilities for the bot."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency during tests
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore

try:  # pragma: no cover - OpenCV may not be available in CI
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Simple representation of a detection result."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


class YoloOnnxDetector:
    """Thin wrapper around an ONNX-exported YOLOv8 model."""

    def __init__(
        self,
        model_path: str,
        class_names: Dict[int, str],
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        if cv2 is None:
            raise RuntimeError("OpenCV is required for preprocessing")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found at '{model_path}'. Place the exported ONNX file "
                "as documented in README.md."
            )

        provider_list = list(providers or ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model_path, providers=provider_list)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [batch, ch, h, w]
        self.class_names = class_names
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        LOGGER.info(
            "Loaded YOLO detector from %s (input=%sx%s)",
            model_path,
            self.input_width,
            self.input_height,
        )

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Resize and pad an image to the network's input dimensions."""

        h0, w0 = image.shape[:2]
        scale = min(self.input_width / w0, self.input_height / h0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        resized = (
            image
            if (new_w, new_h) == (w0, h0)
            else cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        )
        pad_w = self.input_width - new_w
        pad_h = self.input_height - new_h
        dw = pad_w / 2
        dh = pad_h / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        color = (114, 114, 114)
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        img = padded.astype(np.float32)
        img = img[:, :, ::-1]  # BGR -> RGB
        img /= 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img, scale, dw, dh

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """Run the detector on an image and return filtered detections."""

        if image.size == 0:
            return []

        inp, scale, dw, dh = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})[0]
        preds = np.squeeze(outputs, axis=0)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, 0)
        if preds.shape[0] <= preds.shape[1]:
            preds = preds.transpose((1, 0))

        boxes = preds[:, :4]
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        valid = confidences >= conf_threshold
        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]
        if boxes.size == 0:
            return []

        # Convert from center-based xywh to xyxy and rescale to original image.
        x_c = boxes[:, 0]
        y_c = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = (x_c - w / 2 - dw) / scale
        y1 = (y_c - h / 2 - dh) / scale
        x2 = (x_c + w / 2 - dw) / scale
        y2 = (y_c + h / 2 - dh) / scale

        w0, h0 = image.shape[1], image.shape[0]
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w0)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h0)

        keep = self._nms(boxes_xyxy, confidences, class_ids, iou_threshold)
        detections: List[Detection] = []
        for idx in keep:
            cid = int(class_ids[idx])
            detections.append(
                Detection(
                    bbox=tuple(map(float, boxes_xyxy[idx])),
                    confidence=float(confidences[idx]),
                    class_id=cid,
                    label=self.class_names.get(cid, str(cid)),
                )
            )
        return detections

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """Perform class-wise non-maximum suppression."""

        selected: List[int] = []
        for cid in np.unique(class_ids):
            cls_indices = np.where(class_ids == cid)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = confidences[cls_indices]
            order = np.argsort(-cls_scores)
            while order.size > 0:
                idx = order[0]
                selected.append(int(cls_indices[idx]))
                if order.size == 1:
                    break
                rest = order[1:]
                ious = YoloOnnxDetector._iou(cls_boxes[idx], cls_boxes[rest])
                keep = np.where(ious <= iou_threshold)[0]
                order = rest[keep]
        return selected

    @staticmethod
    def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
        if others.size == 0:
            return np.zeros((0,), dtype=np.float32)
        inter_x1 = np.maximum(box[0], others[:, 0])
        inter_y1 = np.maximum(box[1], others[:, 1])
        inter_x2 = np.minimum(box[2], others[:, 2])
        inter_y2 = np.minimum(box[3], others[:, 3])
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        others_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        union = box_area + others_area - inter_area
        return inter_area / np.maximum(union, 1e-6)


def load_detector_settings(config_data: Optional[Dict] = None) -> Dict:
    """Return detector configuration from ``agent_config.yaml``."""

    if config_data is None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "agent_config.yaml")
        if not os.path.exists(config_path):
            return {}
        import yaml

        with open(config_path, "r", encoding="utf-8") as fh:
            config_data = yaml.safe_load(fh) or {}
    return config_data.get("detector", {})


def map_roi_labels(
    roi_names: Iterable[str],
    region_entries: Dict[str, Dict],
    detector_cfg: Optional[Dict] = None,
) -> Dict[str, Dict[str, Optional[Iterable[str]]]]:
    """Build a ROI→detector-target mapping using configuration hints."""

    cfg = detector_cfg or {}
    roi_map = cfg.get("roi_map", {})
    targets: Dict[str, Dict[str, Optional[Iterable[str]]]] = {}
    for name in roi_names:
        entry = region_entries.get(name, {})
        labels: List[str] = []
        if "labels" in entry and isinstance(entry["labels"], list):
            labels.extend(str(v) for v in entry["labels"])
        cfg_entry = roi_map.get(name, {})
        cfg_labels = cfg_entry.get("labels") or cfg_entry.get("classes")
        if isinstance(cfg_labels, (list, tuple, set)):
            labels.extend(str(v) for v in cfg_labels)
        unique_labels = list(dict.fromkeys(labels))
        targets[name] = {"labels": unique_labels or None}
    return targets
