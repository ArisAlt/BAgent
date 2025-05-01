# version: 0.1.3
# path: src/capture_utils.py

import os
import sys

try:
    import win32gui
except ImportError:
    win32gui = None

import cv2
import numpy as np
from PIL import ImageGrab

_window_bbox = None

def capture_screen(select_region: bool = False):
    """
    Capture the EVE client window, or (if select_region) let the user select a sub-region.

    Args:
        select_region: If True, opens a ROI selector on the EVE window.
    Returns:
        If select_region: Tuple[int, int, int, int] of absolute coords.
        Otherwise: np.ndarray BGR image of the EVE window.
    """
    global _window_bbox

    # First-time: determine EVE window bbox
    if _window_bbox is None:
        full = _grab_full_screen_bgr()

        # Try Windows auto-detect
        if win32gui:
            hwnds = []
            win32gui.EnumWindows(lambda h, p: p.append(h) if win32gui.IsWindowVisible(h) and "EVE" in win32gui.GetWindowText(h) else None, hwnds)
            if hwnds:
                x1, y1, x2, y2 = win32gui.GetWindowRect(hwnds[0])
                _window_bbox = (x1, y1, x2, y2)
        # If still unknown, and manual ROI requested:
        if _window_bbox is None:
            if select_region:
                rect = cv2.selectROI("Select EVE Window", full, showCrosshair=False, fromCenter=False)
                cv2.destroyWindow("Select EVE Window")
                x, y, w, h = rect
                _window_bbox = (int(x), int(y), int(x+w), int(y+h))
            else:
                # fallback to full screen
                h, w = full.shape[:2]
                _window_bbox = (0, 0, w, h)

    # Grab the window region
    x1, y1, x2, y2 = _window_bbox
    img_rgb = np.array(ImageGrab.grab(bbox=_window_bbox))
    window_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if select_region:
        # Let user pick a sub-ROI within the window
        rect = cv2.selectROI("Select Region", window_img, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow("Select Region")
        x, y, w, h = rect
        # Convert to absolute coords
        return (int(x1 + x), int(y1 + y), int(x1 + x + w), int(y1 + y + h))

    return window_img

def _grab_full_screen_bgr():
    """Helper: full-screen grab as BGR."""
    img_rgb = np.array(ImageGrab.grab())
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
