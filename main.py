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
    "greencyan": [([40, 80, 60], [92, 255, 255])],    # green ball + cyan "blue" cube
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

# ── Walk-to-object movement settings ──────────────────────────────────
# All tunable. The dog turns to face the object, then walks forward, and stops
# when the object looks big enough (= close).
CENTER_BAND = 0.20    # how centred the object must be before walking forward
TURN_SPEED  = 35      # turn speed deg/s while aiming at the object
TURN_SEARCH = 30      # turn speed deg/s while searching (object not in view)
FWD_STEP    = 12      # forward step size (x translation, max 25)
CLOSE_AREA  = 30000   # stop when the object's area reaches this (= close enough)
                      #   stops too far away?  -> RAISE this
                      #   walks into the object? -> LOWER this
WALK_TIMEOUT = 45     # give up walking after this many seconds

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
# Reject blobs bigger than this fraction of the view = background, not an object.
MAX_AREA_FRAC = 0.35

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


def detect_targets(frame):
    """Return a list of {name, cx, area} for every valid object seen.
    cx is the horizontal offset from centre: -1 = far left, +1 = far right."""
    h, w = frame.shape[:2]
    roi_frame = frame[int(h * SCAN_TOP):h, 0:w]   # full width, table/floor area
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    rh, rw = roi_frame.shape[:2]
    roi_area = rh * rw
    max_obj_area = MAX_AREA_FRAC * roi_area
    out = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = get_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA or area > max_obj_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < MIN_CIRC_GENERIC:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < MIN_SOLIDITY:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            extent = area / (bw * bh) if bw * bh > 0 else 0
            name = classify(color_name, extent)
            if name in VALID_OBJECTS:
                M = cv2.moments(cnt)
                cx_px = M["m10"] / M["m00"] if M["m00"] else rw / 2
                cx = (cx_px - rw / 2) / (rw / 2)
                out.append({"name": name, "cx": cx, "area": area})
    return out


def detect_once(frame):
    """Name of the largest valid object in view, or None."""
    targets = detect_targets(frame)
    if targets:
        targets.sort(key=lambda d: d["area"], reverse=True)
        return targets[0]["name"]
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


def walk_to_object(target, timeout=WALK_TIMEOUT):
    """Turn to face the target, walk toward it, stop when close.
    Returns True if it reached the object, False on timeout."""
    cap = cv2.VideoCapture(0)
    start = time.time()
    last = 0.0
    print(f"  Walking to {target}... (Ctrl+C to stop)")
    try:
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            if time.time() - last < 0.25:
                continue
            last = time.time()

            seen = [d for d in detect_targets(frame) if d["name"] == target]

            if not seen:
                # object not in view → rotate slowly to look for it
                print(f"   searching... (don't see {target})")
                if ROBOT:
                    dog.turn(TURN_SEARCH)
                continue

            obj = max(seen, key=lambda d: d["area"])
            cx, area = obj["cx"], obj["area"]
            print(f"   see {target}: cx={cx:+.2f}  area={area:.0f}")

            # Simulation (no robot): can't physically walk, so just confirm.
            if not ROBOT:
                print(f"  (sim) sees {target} at cx={cx:+.2f}, area={area:.0f}")
                return True

            if area >= CLOSE_AREA:
                dog.stop()
                print("  Arrived at the object.")
                return True

            if cx < -CENTER_BAND:
                dog.turn(TURN_SPEED)        # object is left → turn left
            elif cx > CENTER_BAND:
                dog.turn(-TURN_SPEED)       # object is right → turn right
            else:
                dog.move('x', FWD_STEP)     # centred → walk forward
        return False
    finally:
        cap.release()
        if ROBOT:
            dog.stop()
            time.sleep(0.2)
            dog.reset()
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
            arrived = walk_to_object(target)
            if arrived:
                do_action(ACTIONS[target])
            else:
                print(f"  Couldn't reach {target} in time. Try again.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Goodbye.")
        if ROBOT:
            dog.reset()


if __name__ == "__main__":
    main()
