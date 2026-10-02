"""
Real-Time Drowsiness Detection
OpenCV + MediaPipe Face Mesh + Eye Aspect Ratio (EAR)

Install:
    pip install opencv-python mediapipe numpy

Run:
    python drowsiness_detector.py

Keys:
    q / ESC : quit
    r       : recalibrate (keep your eyes open and look at the camera)
"""

import platform
import subprocess
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CAMERA_INDEX = 0

# Eye landmark indices in MediaPipe Face Mesh, ordered as p1..p6 for the EAR
# formula: p1 = outer/inner corner, p2/p3 = upper lid, p4 = opposite corner,
# p5/p6 = lower lid.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

DEFAULT_EAR_THRESHOLD = 0.21    # used until calibration finishes
CALIBRATION_SECONDS = 3.0       # time spent measuring your open-eye EAR
THRESHOLD_RATIO = 0.75          # alarm threshold = baseline EAR * this ratio
CLOSED_SECONDS_ALERT = 0.7      # eyes closed this long -> drowsiness alert
ALARM_COOLDOWN = 1.0            # minimum seconds between alarm beeps

# ----------------------------------------------------------------------------
# Alarm (non-blocking, cross-platform)
# ----------------------------------------------------------------------------
_alarm_lock = threading.Lock()


def _play_alarm_blocking():
    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            winsound.Beep(1800, 500)
        elif system == "Darwin":
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                check=False,
            )
        else:
            # Linux: try common players, fall back to the terminal bell
            for cmd in (
                ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
            ):
                try:
                    subprocess.run(cmd, check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()


def trigger_alarm():
    """Play the alarm in a background thread so video never freezes."""
    if _alarm_lock.locked():
        return

    def worker():
        with _alarm_lock:
            _play_alarm_blocking()

    threading.Thread(target=worker, daemon=True).start()


# ----------------------------------------------------------------------------
# EAR helpers
# ----------------------------------------------------------------------------
def eye_aspect_ratio(points):
    """
    points: array of shape (6, 2) ordered p1..p6.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def get_eye_points(landmarks, indices, width, height):
    """Convert normalized landmarks to pixel coordinates."""
    return np.array(
        [[landmarks[i].x * width, landmarks[i].y * height] for i in indices],
        dtype=np.float64,
    )


def draw_eye(frame, points, color):
    pts = points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    threshold = DEFAULT_EAR_THRESHOLD
    calibrating = True
    calib_start = time.time()
    calib_samples = []

    closed_since = None      # timestamp when eyes first closed
    last_alarm = 0.0
    drowsy = False

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)

        now = time.time()
        ear = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left = get_eye_points(lm, LEFT_EYE, w, h)
            right = get_eye_points(lm, RIGHT_EYE, w, h)
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

            eye_color = (0, 255, 0)

            if calibrating:
                calib_samples.append(ear)
                elapsed = now - calib_start
                cv2.putText(frame, f"Calibrating... keep eyes open ({elapsed:.1f}/{CALIBRATION_SECONDS:.0f}s)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if elapsed >= CALIBRATION_SECONDS and len(calib_samples) > 10:
                    # Median is robust against any blinks during calibration
                    baseline = float(np.median(calib_samples))
                    threshold = baseline * THRESHOLD_RATIO
                    calibrating = False
                    print(f"Calibration done. Baseline EAR={baseline:.3f}, threshold={threshold:.3f}")
            else:
                if ear < threshold:
                    if closed_since is None:
                        closed_since = now
                    closed_for = now - closed_since
                    eye_color = (0, 165, 255)
                    if closed_for >= CLOSED_SECONDS_ALERT:
                        drowsy = True
                        eye_color = (0, 0, 255)
                        if now - last_alarm >= ALARM_COOLDOWN:
                            trigger_alarm()
                            last_alarm = now
                else:
                    closed_since = None
                    drowsy = False

            draw_eye(frame, left, eye_color)
            draw_eye(frame, right, eye_color)
        else:
            # No face: reset the timer so we don't alert on a missing face
            closed_since = None
            drowsy = False
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---- HUD ----
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}  (thr {threshold:.3f})", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if drowsy:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(frame, "DROWSINESS ALERT!", (int(w * 0.5) - 190, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            calibrating = True
            calib_start = time.time()
            calib_samples = []
            closed_since = None
            drowsy = False

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()