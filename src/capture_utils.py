# version: 0.8.4
# path: src/capture_utils.py

import cv2
import numpy as np
import ctypes
from ctypes.wintypes import RECT

try:
    from .logger import get_logger  # package import
except ImportError:  # pragma: no cover - fallback for direct execution
    from logger import get_logger

try:
    import win32gui
    import win32ui
    import win32con
except Exception:  # pragma: no cover - allow headless testing
    win32gui = win32ui = win32con = None
import time

logger = get_logger(__name__)

# Make the process DPI aware so coordinates match real pixels (Windows only)
if hasattr(ctypes, "windll") and win32gui is not None:
    ctypes.windll.user32.SetProcessDPIAware()
_first_foreground = True


def get_window_rect(title: str):
    hwnd = ctypes.windll.user32.FindWindowW(0, title)
    if hwnd == 0:
        raise RuntimeError(f"[Capture] ❌ Window '{title}' not found.")
    rect = RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
    return (rect.left, rect.top, rect.right, rect.bottom), hwnd


def capture_screen(select_region=False, window_title=None):
    """
    Capture a window using PrintWindow (safest GDI method for layered content).
    Falls back to `pyautogui.screenshot()` or Pillow's `ImageGrab.grab()` if
    PrintWindow fails or returns a white image. Works best when EVE is not
    fullscreen-exclusive.
    """
    global _first_foreground

    if window_title is None:
        try:
            from .config import get_window_title
        except Exception:  # pragma: no cover - fallback for direct execution
            from config import get_window_title

        window_title = get_window_title()

    if win32gui is None:
        # headless fallback for tests
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        if select_region:
            roi = cv2.selectROI(
                "Select Region", img, showCrosshair=True, fromCenter=False
            )
            if hasattr(cv2, "destroyWindow"):
                try:
                    cv2.destroyWindow("Select Region")
                except Exception:
                    pass
            x, y, w, h = roi
            if w == 0 or h == 0:
                return None
            return (x, y, x + w, y + h)
        return img

    try:
        (left, top, right, bottom), hwnd = get_window_rect(window_title)
    except RuntimeError as e:
        logger.error(str(e))
        return None

    width = right - left
    height = bottom - top

    # Bring EVE to foreground on first run
    if _first_foreground and win32gui is not None:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"[Capture] Could not bring to foreground: {e}")
        _first_foreground = False

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bmp)

    img_bgr = None
    used_method = "PrintWindow"

    try:
        result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
    except Exception as e:
        logger.warning(f"[Capture] PrintWindow error: {e}")
        result = 0

    if result == 1:
        bmp_str = bmp.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype="uint8").reshape((height, width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if np.mean(img_bgr) > 250:
            logger.warning("[Capture] PrintWindow returned white image.")
            img_bgr = None
    else:
        logger.warning("[Capture] PrintWindow failed.")

    # Cleanup
    win32gui.DeleteObject(bmp.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    if img_bgr is None:
        try:
            import pyautogui
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            used_method = "pyautogui"
        except Exception as e:
            logger.warning(f"[Capture] pyautogui.screenshot failed: {e}")
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
                img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                used_method = "ImageGrab"
            except Exception as e:
                logger.error(f"[Capture] ImageGrab failed: {e}")
                return None

    logger.info(f"[Capture] captured via {used_method}")

    if select_region:
        roi = cv2.selectROI(
            "Select Region", img_bgr, showCrosshair=True, fromCenter=False
        )
        if hasattr(cv2, "destroyWindow"):
            try:
                cv2.destroyWindow("Select Region")
            except Exception:
                pass
        x, y, w, h = roi
        if w == 0 or h == 0:
            return None
        return (x, y, x + w, y + h)

    return img_bgr


# --- Test directly ---
if __name__ == "__main__":
    from .config import get_window_title

    frame = capture_screen(window_title=get_window_title())
    if frame is not None:
        cv2.imshow("PrintWindow Capture", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.error("Could not capture.")
