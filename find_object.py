"""
find_object.py  —  LAYER 1: find an object on the floor
========================================================
Unlike detect_object.py (which assumes you HOLD the object in the centre),
this scans a WIDE area and reports WHERE the object is, so the robot can
later turn toward it and walk over.

For each detection it gives:
    name   : which object (e.g. 'red_ball')
    cx     : horizontal position, -1.0 (far left) .. 0.0 (centre) .. +1.0 (far right)
    area   : blob size in pixels  (bigger = closer)
    side   : 'LEFT' / 'CENTRE' / 'RIGHT'  (handy for steering)

Run it on the robot, put objects on the floor in view, and read the numbers.
Tune SCAN_TOP and MIN_AREA to your robot camera (notes below).
"""

import cv2
import numpy as np
import time

# ── Colour ranges (HSV) — same as the working detector ────────────────
COLOR_RANGES = {
    "red":    [([0, 150, 80], [6, 255, 255]), ([170, 150, 80], [180, 255, 255])],
    "green":  [([35, 60, 50],   [85, 255, 255])],
    "blue":   [([100, 50, 50],  [130, 255, 255])],
    "purple": [([140, 50, 100], [170, 255, 255])],
    "yellow": [([18, 120, 120], [32, 255, 255])],
}
COLOR_SHAPE = {"red": "ball", "green": "ball", "yellow": "cube",
               "purple": "cube", "blue": None}
BLUE_BALL_CIRC = 0.77

ACTIONS = {
    "blue_ball": "STAND TALL", "blue_cube": "SIT DOWN", "green_ball": "LIE DOWN",
    "purple_cube": "WAVE ARM", "red_ball": "ATTACK + BARK", "yellow_cube": "SPIN / DANCE",
}
VALID_OBJECTS = set(ACTIONS.keys())

# ── Tunables for the FLOOR view (adjust on the robot) ─────────────────
# Look at the lower part of the frame where the floor is. 0.0 = whole frame,
# 0.35 = ignore the top 35% (wall/horizon). Start at 0.25.
SCAN_TOP = 0.25
# Far objects are small, so this is LOW. If background noise sneaks in,
# raise it; if it misses distant objects, lower it.
MIN_AREA = 1500
MAX_AREA = 250000
MIN_CIRC = {"green": 0.45, "red": 0.25, "yellow": 0.25, "purple": 0.25, "blue": 0.25}
CENTRE_BAND = 0.15   # |cx| below this counts as "centred"


def get_mask(hsv, ranges):
    combined = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined = m if combined is None else cv2.bitwise_or(combined, m)
    return combined


def find_objects(frame):
    """Return a list of dicts for every valid object found in the floor area."""
    h, w = frame.shape[:2]
    y_start = int(h * SCAN_TOP)
    roi = frame[y_start:h, 0:w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    found = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = get_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            if circ < MIN_CIRC.get(color_name, 0.25):
                continue

            fixed = COLOR_SHAPE[color_name]
            if fixed is None:  # blue
                shape = "ball" if circ >= BLUE_BALL_CIRC else "cube"
            else:
                shape = fixed
            name = f"{color_name}_{shape}"
            if name not in VALID_OBJECTS:
                continue

            M = cv2.moments(cnt)
            cx_px = M["m10"] / M["m00"] if M["m00"] else w / 2
            cx = (cx_px - w / 2) / (w / 2)        # -1 .. +1
            side = "CENTRE" if abs(cx) < CENTRE_BAND else ("LEFT" if cx < 0 else "RIGHT")
            found.append({"name": name, "cx": cx, "area": area, "side": side})

    found.sort(key=lambda d: d["area"], reverse=True)
    return found


def find_target(frame, target):
    """Return the best detection matching `target`, or None."""
    for obj in find_objects(frame):
        if obj["name"] == target:
            return obj
    return None


def main():
    print("Floor object finder. Put objects on the floor in view. Ctrl+C to quit.\n")
    cap = cv2.VideoCapture(0)
    last = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if time.time() - last < 0.4:
                continue
            last = time.time()
            objs = find_objects(frame)
            if objs:
                for o in objs[:3]:
                    print(f"  {o['name']:<12} side={o['side']:<6} cx={o['cx']:+.2f} area={o['area']:.0f}")
                print("  " + "-" * 30)
            else:
                print("  (nothing found)")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("\nstopped")


if __name__ == "__main__":
    main()
