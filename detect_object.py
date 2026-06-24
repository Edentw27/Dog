"""
detect_object.py
-----------------
Standalone tester for the RoboDog vision system.

Pure computer vision: HSV colour masking + contour analysis.

Key idea: of the 6 objects, only BLUE comes in two shapes (ball + cube).
Red/green = ball, yellow/purple = cube. So colour alone identifies four of
them; shape detection only runs for blue.
"""

import cv2
import numpy as np
import time
from collections import Counter

# ── Colour ranges (HSV) ───────────────────────────────────────────────
COLOR_RANGES = {
    "red":    [([0, 150, 80], [6, 255, 255]), ([170, 150, 80], [180, 255, 255])],
    "green":  [([35, 60, 50],   [85, 255, 255])],
    "blue":   [([100, 50, 50],  [130, 255, 255])],
    "purple": [([140, 50, 100], [170, 255, 255])],
    "yellow": [([18, 120, 120], [32, 255, 255])],
}

# Each colour maps to a fixed shape — EXCEPT blue, which needs geometry.
COLOR_SHAPE = {
    "red":    "ball",
    "green":  "ball",
    "yellow": "cube",
    "purple": "cube",
    "blue":   None,     # decide ball vs cube from aspect ratio
}

# Minimum circularity to accept a blob as a real object (kills stringy
# background noise). Green is stricter because the desk/wall mimics it.
MIN_CIRC = {"green": 0.55, "red": 0.30, "yellow": 0.30, "purple": 0.30, "blue": 0.30}

ACTIONS = {
    "blue_ball":   "STAND TALL",
    "blue_cube":   "SIT DOWN",
    "green_ball":  "LIE DOWN",
    "purple_cube": "WAVE ARM",
    "red_ball":    "ATTACK + BARK",
    "yellow_cube": "SPIN / DANCE",
}

VALID_OBJECTS = set(ACTIONS.keys())
MIN_AREA = 3000
MAX_AREA = 250000
BLUE_BALL_CIRC = 0.77       # blue: circularity at/above this = ball, below = cube


def get_mask(hsv, ranges):
    combined = None
    for (lower, upper) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined = mask if combined is None else cv2.bitwise_or(combined, mask)
    return combined


def detect_once(frame):
    h, w = frame.shape[:2]
    roi_frame = frame[int(h * 0.20):int(h * 0.80), int(w * 0.30):int(w * 0.70)]
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    results = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = get_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < MIN_CIRC.get(color_name, 0.30):
                continue  # reject background noise

            fixed_shape = COLOR_SHAPE[color_name]
            if fixed_shape is None:  # blue → decide ball vs cube by roundness
                shape = "ball" if circularity >= BLUE_BALL_CIRC else "cube"
            else:
                shape = fixed_shape

            obj_name = f"{color_name}_{shape}"
            if obj_name in VALID_OBJECTS:
                results.append((obj_name, area))

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0]
    return None


def main():
    print("\n" + "=" * 50)
    print("  ROBODOG - Color + Shape Detection")
    print("  Hold object in CENTER of camera view.")
    print("  Press CTRL+C to quit.")
    print("=" * 50 + "\n")

    cap = cv2.VideoCapture(0)
    readings = []
    last_check = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if time.time() - last_check < 0.3:
                continue
            last_check = time.time()
            result = detect_once(frame)
            readings.append(result if result else "none")
            if len(readings) >= 5:
                majority = Counter(readings).most_common(1)[0][0]
                readings = []
                if majority != "none":
                    print(f"  DETECTED: {majority:<15} | Action: {ACTIONS.get(majority, '?')}")
                else:
                    print("  Waiting...")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
