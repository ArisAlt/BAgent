# version: 0.8.0
# path: src/capture_utils.py

import cv2
import numpy as np
import ctypes
from ctypes.wintypes import RECT
try:
    import win32gui
    import win32ui
    import win32con
except Exception:  # pragma: no cover - allow headless testing
    win32gui = win32ui = win32con = None
import time

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

def capture_screen(select_region=False, window_title="EVE - CitizenZero"):
    """
    Capture a window using PrintWindow (safest GDI method for layered content).
    Works best when EVE is not fullscreen-exclusive.
    """
    global _first_foreground

    if win32gui is None:
        # headless fallback for tests
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        if select_region:
            roi = cv2.selectROI("Select Region", img, showCrosshair=True, fromCenter=False)
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
        print(str(e))
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
            print(f"[Capture] ⚠️ Could not bring to foreground: {e}")
        _first_foreground = False

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bmp     = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bmp)

    # Try PrintWindow
    result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
    if result != 1:
        print("[Capture] ⚠️ PrintWindow failed. Screen may be black.")
        return None

    bmp_info = bmp.GetInfo()
    bmp_str = bmp.GetBitmapBits(True)
    img = np.frombuffer(bmp_str, dtype='uint8').reshape((height, width, 4))

    # Cleanup
    win32gui.DeleteObject(bmp.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if select_region:
        roi = cv2.selectROI("Select Region", img_bgr, showCrosshair=True, fromCenter=False)
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
    frame = capture_screen(window_title="EVE - CitizenZero")
    if frame is not None:
        cv2.imshow("PrintWindow Capture", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Could not capture.")
