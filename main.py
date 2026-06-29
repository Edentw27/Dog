"""
main.py  —  RoboDog full pipeline
==================================

    voice command  ->  parse colour+shape  ->  camera scan  ->  robot action

Runs in TWO modes automatically:
  * On your laptop (no xgolib)  -> SIMULATION: prints the action.
  * On the robot   (xgolib ok)  -> REAL: moves the dog via dog.action(id).

Vision: pure HSV colour + contour analysis (no trained model needed).
Voice:  Google Speech Recognition via the SpeechRecognition library.
"""

import cv2
import numpy as np
import time
import os
from collections import Counter
import speech_recognition as sr

# ── Try to connect to the robot. If unavailable, run in simulation. ────
try:
    from xgolib import XGO
    dog = XGO('xgomini')
    dog.reset()
    ROBOT = True
    print("[robot] xgolib connected — REAL mode.")
except Exception as e:
    dog = None
    ROBOT = False
    print(f"[robot] xgolib not available — SIMULATION mode ({e}).")

# ── Colour ranges (HSV) ───────────────────────────────────────────────
# Your "blue" cube reads as cyan/green to the camera (H~56-75), the SAME zone
# as the green ball. So green ball and blue cube can't be told apart by colour —
# they're separated by SHAPE instead (ball = round, cube = square). The true
# blue BALL reads as real blue (H~100+), so it gets its own range.
COLOR_RANGES = {
    "red":       [([0, 150, 80], [6, 255, 255]), ([170, 150, 80], [180, 255, 255])],
    "yellow":    [([18, 120, 120], [32, 255, 255])],
    "purple":    [([140, 50, 100], [170, 255, 255])],
    "greencyan": [([38, 55, 45], [92, 255, 255])],    # green ball + cyan "blue" cube
    "trueblue":  [([95, 60, 45], [132, 255, 255])],   # real blue ball
}

# ── Object -> action name ─────────────────────────────────────────────
ACTIONS = {
    "blue_ball":   "STAND TALL",
    "blue_cube":   "SIT DOWN",
    "green_ball":  "LIE DOWN",
    "purple_cube": "WAVE ARM",
    "red_ball":    "ATTACK + BARK",
    "yellow_cube": "SPIN / DANCE",
}

# ── Action name -> xgolib preset action id ────────────────────────────
# IMPORTANT: these ids are STARTING GUESSES based on the standard XGO
# preset table. CONFIRM each one on your robot with scripts/explore.py,
# then correct the numbers here.
ACTION_IDS = {
    "LIE DOWN":      1,    # 1 = get down            (confirmed in docs)
    "STAND TALL":    2,    # 2 = stand up            (confirmed in docs)
    "SIT DOWN":      12,   # 12 = sit down           (was 6 = squat — WRONG)
    "WAVE ARM":      13,   # 13 = wave               (was 17 = seeking food — WRONG)
    "ATTACK + BARK": 16,   # 16 = swing L/R — no real "attack" preset; CONFIRM on robot,
                           #      or replace with a scripted lunge (see note in chat)
    "SPIN / DANCE":  4,    # 4 = circle (clean spin); try 10 = 3-axis rotation for "dance"
}

# How long (seconds) to let a preset action play before resetting to neutral.
# Presets take roughly 3-6s. If moves still get cut off, raise this. If the dog
# pauses too long between actions, lower it.
ACTION_TIME = 4.0

# ── Shape decision ────────────────────────────────────────────────────
# A ball is round, a cube is square. We tell them apart by EXTENT = how much
# of the bounding box the blob fills. A circle fills ~0.78 of its box; a square
# fills ~1.0. So below the threshold = ball (round), at/above = cube (square).
# Your blue ball measured ext~0.75. If a cube gets called a ball, raise this;
# if a ball gets called a cube, lower it.
SHAPE_EXTENT = 0.80

# Reject noise: must be reasonably compact and solid (not a stringy background).
MIN_CIRC_GENERIC = 0.40
MIN_SOLIDITY = 0.85

VALID_COLORS = ["red", "green", "blue", "yellow", "purple"]
VALID_SHAPES = ["ball", "cube"]
VALID_OBJECTS = set(ACTIONS.keys())
# Scan area: from SCAN_TOP down to the bottom, full width — so an object on the
# table (off-centre or far) is still inside the search region.
SCAN_TOP = 0.20
MIN_AREA = 400             # lowered so far-away (smaller) objects still pass
MAX_AREA = 250000

# Path to a bark sound for the red ball (optional). Put a wav next to this file.
BARK_WAV = os.path.join(os.path.dirname(__file__), "bark.wav")


# ── Vision ────────────────────────────────────────────────────────────
def get_mask(hsv, ranges):
    combined = None
    for (lower, upper) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined = mask if combined is None else cv2.bitwise_or(combined, mask)
    return combined


def classify(color_name, extent):
    """Turn a (colour, shape) pair into one of the six object names.
    red/yellow/purple are unambiguous. green-vs-blue-cube and blue-ball-vs-cube
    are decided by EXTENT (round = ball, square = cube)."""
    if color_name == "red":
        return "red_ball"
    if color_name == "yellow":
        return "yellow_cube"
    if color_name == "purple":
        return "purple_cube"
    if color_name == "greencyan":
        # green ball (round) vs the cyan "blue" cube (square)
        return "green_ball" if extent < SHAPE_EXTENT else "blue_cube"
    if color_name == "trueblue":
        return "blue_ball" if extent < SHAPE_EXTENT else "blue_cube"
    return None


def detect_once(frame):
    h, w = frame.shape[:2]
    roi_frame = frame[int(h * SCAN_TOP):h, 0:w]   # full width, table area
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    results = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = get_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < MIN_CIRC_GENERIC:
                continue  # reject stringy background noise

            # Solidity: real objects are solid/compact (~0.9+); background
            # false-positives are ragged. This stops empty-table false hits.
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < MIN_SOLIDITY:
                continue

            # Extent = blob area / bounding-box area → round vs square.
            x, y, bw, bh = cv2.boundingRect(cnt)
            extent = area / (bw * bh) if bw * bh > 0 else 0

            obj_name = classify(color_name, extent)
            if obj_name in VALID_OBJECTS:
                results.append((obj_name, area))

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0]
    return None


def scan_for_object(target=None, votes=5, timeout=20):
    cap = cv2.VideoCapture(0)
    readings = []
    last_check = time.time()
    start = time.time()
    print("  Scanning... (hold the object in the centre of the frame)")
    try:
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            if time.time() - last_check < 0.3:
                continue
            last_check = time.time()
            result = detect_once(frame)
            readings.append(result if result else "none")
            if len(readings) >= votes:
                majority = Counter(readings).most_common(1)[0][0]
                readings = []
                if majority != "none":
                    if target is None or majority == target:
                        return majority
                    print(f"  (saw {majority}, still looking for {target})")
    finally:
        cap.release()
    return None


# ── Voice ─────────────────────────────────────────────────────────────
def parse_command(text):
    text = text.lower()
    color = next((c for c in VALID_COLORS if c in text), None)
    shape = next((s for s in VALID_SHAPES if s in text), None)
    if color and shape:
        return f"{color}_{shape}"
    return None


def listen_for_command(timeout=6):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("\n  Listening... say e.g. 'fetch green ball'")
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=4)
        except sr.WaitTimeoutError:
            print("  (heard nothing)")
            return None
    try:
        text = r.recognize_google(audio)
        print(f"  Heard: \"{text}\"")
        return parse_command(text)
    except sr.UnknownValueError:
        print("  (couldn't understand)")
    except sr.RequestError as e:
        print(f"  Speech service error: {e}")
    return None


# ── Sound + robot action ──────────────────────────────────────────────
def play_bark():
    if not os.path.exists(BARK_WAV):
        print("  (woof! — no bark.wav found)")
        return
    # try a few players so it works on Mac and on the Pi
    for cmd in (f"afplay '{BARK_WAV}'", f"aplay '{BARK_WAV}'"):
        if os.system(cmd + " 2>/dev/null") == 0:
            return
    print("  (couldn't play bark.wav)")


def do_action(action):
    print(f"  >>> ACTION: {action}")

    if action == "ATTACK + BARK":
        play_bark()

    if not ROBOT:
        return  # simulation mode: nothing physical to do

    action_id = ACTION_IDS.get(action)
    if action_id is None:
        print(f"  (no action id mapped for {action})")
        return
    try:
        dog.action(action_id)      # wait=True doesn't block in xgolib 0.3.1
        time.sleep(ACTION_TIME)    # let the preset finish before resetting
        dog.reset()
    except Exception as e:
        print(f"  robot error: {e}")


# ── Main loop ─────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 50)
    print("  ROBODOG  —  voice -> vision -> action")
    print(f"  Mode: {'REAL ROBOT' if ROBOT else 'SIMULATION'}")
    print("  Ctrl+C to quit.")
    print("=" * 50)
    try:
        while True:
            target = listen_for_command()
            if not target:
                print("  Try again — e.g. 'fetch blue cube'.")
                continue
            print(f"  Target: {target}  (action: {ACTIONS.get(target, '?')})")
            found = scan_for_object(target=target)
            if found:
                print(f"  FOUND {found}!")
                do_action(ACTIONS[found])
            else:
                print(f"  Didn't find {target} in time. Try again.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Goodbye.")
        if ROBOT:
            dog.reset()


if __name__ == "__main__":
    main()
