# version: 0.3.0
# path: src/roi_live_overlay.py

import cv2
import time
from roi_capture import RegionHandler
from capture_utils import capture_screen

def draw_overlay(rois, screen):
    for name, (x1, y1, x2, y2) in rois.items():
        color = (0, 255, 0)
        cv2.rectangle(screen, (x1, y1), (x2, y2), color, 2)
        cv2.putText(screen, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return screen

def main_loop():
    rh = RegionHandler()
    print("[Overlay] Press ESC to exit")

    while True:
        screen = capture_screen()
        if screen is None:
            print("[Overlay] ❌ Screen capture failed.")
            continue

        rois = {}
        for name in rh.list_regions():
            coords = rh.load(name)
            if coords:
                rois[name] = coords

        frame = draw_overlay(rois, screen.copy())
        cv2.imshow("🛰 ROI Live Overlay", frame)

        if cv2.waitKey(30) == 27:  # ESC
            break
        time.sleep(0.15)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
