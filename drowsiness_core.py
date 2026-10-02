"""
drowsiness_core.py
Shared detection engine used by both the desktop app (run_desktop.py)
and the Streamlit dashboard (app.py).

Detects:
  * Eye closure      -> Eye Aspect Ratio (EAR)
  * Yawning          -> Mouth Aspect Ratio (MAR)
  * Head nodding     -> head pitch via solvePnP
  * Looking away     -> head yaw via solvePnP
and logs every alert event to a CSV file.

Requires: opencv-python, mediapipe==0.10.14, numpy
"""

import csv
import math
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# Landmark indices (MediaPipe Face Mesh)
# ----------------------------------------------------------------------------
# Eyes, ordered p1..p6 for EAR: corner, upper, upper, corner, lower, lower
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth (inner lips): 3 vertical pairs, then left/right corners
#            up1  lo1  up2  lo2  up3  lo3  left right
MOUTH = [82, 87, 13, 14, 312, 317, 78, 308]
MOUTH_OUTLINE = [78, 82, 13, 312, 308, 317, 14, 87]

# Head pose points (frame is mirrored, so subject's left side is image-left)
# nose tip, chin, eye outer (image-left), eye outer (image-right),
# mouth corner (image-left), mouth corner (image-right)
POSE_IDX = [1, 152, 263, 33, 291, 61]

# Generic 3D face model in camera-style axes (x right, y down, z away from camera)
POSE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, 330.0, 65.0),     # chin
        (-225.0, -170.0, 135.0),  # eye outer corner, image-left
        (225.0, -170.0, 135.0),   # eye outer corner, image-right
        (-150.0, 150.0, 125.0),   # mouth corner, image-left
        (150.0, 150.0, 125.0),    # mouth corner, image-right
    ],
    dtype=np.float64,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "drowsiness_log.csv"
)


@dataclass
class Config:
    camera_index: int = 0
    calibration_seconds: float = 3.0

    # Eyes
    default_ear_threshold: float = 0.21   # used until calibration finishes
    threshold_ratio: float = 0.75         # alarm EAR = baseline EAR * ratio
    eye_closed_seconds: float = 0.7

    # Yawn
    mar_threshold: float = 0.60
    yawn_seconds: float = 1.2

    # Head pose (degrees, measured relative to your calibrated neutral pose)
    nod_degrees: float = 15.0             # head dropping forward/down
    nod_seconds: float = 1.0
    away_degrees: float = 35.0            # head turned left/right
    away_seconds: float = 2.0

    # Fatigue score (PERCLOS + recent events)
    perclos_window: float = 60.0          # seconds used for PERCLOS
    score_window: float = 300.0           # seconds of event history for the score
    fatigue_medium: float = 30.0          # score >= this -> MEDIUM
    fatigue_high: float = 60.0            # score >= this -> HIGH

    # Alarm / logging
    sound: bool = True
    alarm_cooldown: float = 1.5
    log_path: str = DEFAULT_LOG_PATH


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
            subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], check=False)
        else:
            for cmd in (
                ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
            ):
                try:
                    subprocess.run(cmd, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()


def trigger_alarm():
    """Play the alarm on a background thread so video never freezes."""
    if _alarm_lock.locked():
        return

    def worker():
        with _alarm_lock:
            _play_alarm_blocking()

    threading.Thread(target=worker, daemon=True).start()


# ----------------------------------------------------------------------------
# CSV event logger
# ----------------------------------------------------------------------------
class EventLogger:
    HEADER = ["timestamp", "event", "duration_s", "ear", "mar", "pitch_deg", "yaw_deg"]

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.HEADER)
        except OSError as e:
            print(f"[log] Could not create {path}: {e}")

    def log(self, event, duration, ear, mar, pitch, yaw):
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event,
            round(duration, 2),
            round(ear, 3),
            round(mar, 3),
            round(pitch, 1),
            round(yaw, 1),
        ]
        try:
            with self._lock, open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except PermissionError:
            print("[log] Cannot write to the CSV. Is it open in Excel? Close it and retry.")
        except OSError as e:
            print(f"[log] Write failed: {e}")


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def _pts(landmarks, indices, w, h):
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float64
    )


def eye_aspect_ratio(p):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz == 0:
        return 0.0
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horiz)


def mouth_aspect_ratio(p):
    """MAR = (three vertical lip distances) / (3 * mouth width)"""
    horiz = np.linalg.norm(p[6] - p[7])
    if horiz == 0:
        return 0.0
    vert = (
        np.linalg.norm(p[0] - p[1])
        + np.linalg.norm(p[2] - p[3])
        + np.linalg.norm(p[4] - p[5])
    )
    return vert / (3.0 * horiz)


def _wrap(angle):
    """Wrap an angle in degrees into [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def _euler_from_rmat(R):
    """Pitch, yaw, roll in degrees, each guaranteed to be within [-180, 180]."""
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def head_pose(landmarks, w, h, prev=None):
    """
    Returns (pitch, yaw, roll, pose_state) in degrees, or None if the pose
    could not be solved or looks implausible (flipped solution).
    pitch > 0 : head tilted down (nodding)
    yaw       : head turned left/right
    `prev` is the pose_state from the previous frame; reusing it keeps the
    solver on the same solution instead of jumping to a mirrored one.
    """
    img_pts = _pts(landmarks, POSE_IDX, w, h)
    focal = float(w)
    cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    if prev is not None:
        ok, rvec, tvec = cv2.solvePnP(
            POSE_MODEL, img_pts, cam, dist,
            rvec=prev[0].copy(), tvec=prev[1].copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            POSE_MODEL, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _euler_from_rmat(rmat)

    # A real head never pitches/yaws beyond ~80 deg while still facing the
    # camera; larger values mean the solver picked a flipped solution.
    if abs(pitch) > 80 or abs(yaw) > 80 or tvec[2][0] <= 0:
        return None
    return pitch, yaw, roll, (rvec, tvec)


def _ema(prev, new, alpha=0.4):
    return new if prev is None else alpha * new + (1 - alpha) * prev


# ----------------------------------------------------------------------------
# Episode timer: "has this condition been continuously true long enough?"
# ----------------------------------------------------------------------------
class Episode:
    def __init__(self):
        self.since = None
        self.fired = False

    def reset(self):
        self.since = None
        self.fired = False

    def update(self, active, now, min_seconds):
        """Returns (duration, just_fired)."""
        if not active:
            self.reset()
            return 0.0, False
        if self.since is None:
            self.since = now
        duration = now - self.since
        just_fired = duration >= min_seconds and not self.fired
        if just_fired:
            self.fired = True
        return duration, just_fired


# ----------------------------------------------------------------------------
# Per-frame result
# ----------------------------------------------------------------------------
@dataclass
class FrameState:
    face: bool = False
    calibrating: bool = True
    calib_progress: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0   # relative to calibrated neutral, + = down
    yaw: float = 0.0     # relative to calibrated neutral
    threshold: float = 0.0
    eyes_alert: bool = False
    yawn_alert: bool = False
    nod_alert: bool = False
    away_alert: bool = False
    perclos: float = 0.0          # fraction 0..1 of the last minute with eyes closed
    fatigue_score: float = 0.0    # 0..100
    fatigue_level: str = "LOW"    # LOW / MEDIUM / HIGH
    counts: dict = field(default_factory=dict)
    new_events: list = field(default_factory=list)

    @property
    def any_alert(self):
        return self.eyes_alert or self.yawn_alert or self.nod_alert or self.away_alert


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
class DrowsinessDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.logger = EventLogger(self.cfg.log_path)

        self.ep_eyes = Episode()
        self.ep_yawn = Episode()
        self.ep_nod = Episode()
        self.ep_away = Episode()

        self.counts = {"DROWSY_EYES": 0, "YAWN": 0, "HEAD_NOD": 0, "LOOKING_AWAY": 0}
        self.last_alarm = 0.0

        self.baseline_ear = None
        self.base_pitch = 0.0
        self.base_yaw = 0.0
        self._pitch_s = None
        self._yaw_s = None
        self._pose_state = None

        # fatigue tracking
        self._hist = deque()          # (timestamp, dt, eyes_closed)
        self._cl = 0.0
        self._tot = 0.0
        self._last_t = None
        self.recent_events = deque()  # (timestamp, event_name)
        self.fatigue_level = "LOW"
        self.perclos = 0.0
        self.fatigue_score = 0.0

        self.recalibrate()

    # ---- public helpers ----------------------------------------------------
    @property
    def threshold(self):
        if self.baseline_ear is None:
            return self.cfg.default_ear_threshold
        return self.baseline_ear * self.cfg.threshold_ratio

    def recalibrate(self):
        self.calibrating = True
        self.calib_start = time.time()
        self.calib_samples = []
        self._reset_fatigue()
        self.reset_timers()

    def reset_timers(self):
        for ep in (self.ep_eyes, self.ep_yawn, self.ep_nod, self.ep_away):
            ep.reset()
        self._last_t = None

    def _reset_fatigue(self):
        self._hist.clear()
        self._cl = self._tot = 0.0
        self.recent_events.clear()
        self.fatigue_level = "LOW"
        self.perclos = self.fatigue_score = 0.0

    def _update_fatigue(self, now, dt, eyes_closed):
        """Update PERCLOS and the 0-100 fatigue score. Returns True if it just became HIGH."""
        cfg = self.cfg
        # PERCLOS: share of the last `perclos_window` seconds with eyes closed
        self._hist.append((now, dt, eyes_closed))
        self._tot += dt
        if eyes_closed:
            self._cl += dt
        while self._hist and now - self._hist[0][0] > cfg.perclos_window:
            _, d0, c0 = self._hist.popleft()
            self._tot -= d0
            if c0:
                self._cl -= d0
        self.perclos = max(0.0, self._cl) / self._tot if self._tot > 1.0 else 0.0

        # recent discrete events
        while self.recent_events and now - self.recent_events[0][0] > cfg.score_window:
            self.recent_events.popleft()
        n_eyes = sum(1 for _, e in self.recent_events if e == "DROWSY_EYES")
        n_yawn = sum(1 for _, e in self.recent_events if e == "YAWN")
        n_nod = sum(1 for _, e in self.recent_events if e == "HEAD_NOD")

        # weights: PERCLOS 50, microsleeps 20, yawns 20, nods 10  (max 100)
        self.fatigue_score = (
            50.0 * min(self.perclos / 0.25, 1.0)
            + 20.0 * min(n_eyes / 2.0, 1.0)
            + 20.0 * min(n_yawn / 3.0, 1.0)
            + 10.0 * min(n_nod / 2.0, 1.0)
        )

        # level with hysteresis so it doesn't flicker around the HIGH limit
        prev = self.fatigue_level
        if self.fatigue_score >= cfg.fatigue_high:
            level = "HIGH"
        elif prev == "HIGH" and self.fatigue_score >= cfg.fatigue_high - 10:
            level = "HIGH"
        elif self.fatigue_score >= cfg.fatigue_medium:
            level = "MEDIUM"
        else:
            level = "LOW"
        self.fatigue_level = level
        return level == "HIGH" and prev != "HIGH"

    def close(self):
        self.face_mesh.close()

    # ---- main entry --------------------------------------------------------
    def process(self, frame_bgr):
        """Takes a raw BGR webcam frame. Returns (annotated BGR frame, FrameState)."""
        cfg = self.cfg
        frame = cv2.flip(frame_bgr, 1)  # mirror view
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        now = time.time()

        s = FrameState(calibrating=self.calibrating, threshold=self.threshold,
                       counts=dict(self.counts), perclos=self.perclos,
                       fatigue_score=self.fatigue_score,
                       fatigue_level=self.fatigue_level)

        if not results.multi_face_landmarks:
            self.reset_timers()
            self._pitch_s = self._yaw_s = None
            self._pose_state = None
            self._draw_hud(frame, s, None)
            return frame, s

        lm = results.multi_face_landmarks[0].landmark
        left = _pts(lm, LEFT_EYE, w, h)
        right = _pts(lm, RIGHT_EYE, w, h)
        mouth = _pts(lm, MOUTH, w, h)

        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        pose = head_pose(lm, w, h, self._pose_state)
        if pose is not None:
            self._pitch_s = _ema(self._pitch_s, pose[0])
            self._yaw_s = _ema(self._yaw_s, pose[1])
            self._pose_state = pose[3]
        else:
            self._pose_state = None  # force a fresh solve next frame
        pitch = self._pitch_s if self._pitch_s is not None else 0.0
        yaw = self._yaw_s if self._yaw_s is not None else 0.0

        s.face = True
        s.ear, s.mar = ear, mar

        # ---------------- Calibration ----------------
        if self.calibrating:
            self.calib_samples.append((ear, pitch, yaw))
            elapsed = now - self.calib_start
            s.calib_progress = min(elapsed / cfg.calibration_seconds, 1.0)
            if elapsed >= cfg.calibration_seconds and len(self.calib_samples) >= 10:
                arr = np.array(self.calib_samples)
                self.baseline_ear = float(np.median(arr[:, 0]))
                self.base_pitch = float(np.median(arr[:, 1]))
                self.base_yaw = float(np.median(arr[:, 2]))
                self.calibrating = False
                s.calibrating = False
                s.threshold = self.threshold
                print(f"Calibration done. Baseline EAR={self.baseline_ear:.3f}, "
                      f"threshold={self.threshold:.3f}, "
                      f"neutral pitch={self.base_pitch:.1f}, yaw={self.base_yaw:.1f}")
            s.pitch, s.yaw = 0.0, 0.0
            self._draw_hud(frame, s, (left, right, mouth))
            return frame, s

        # ---------------- Detection ----------------
        pitch_dev = _wrap(pitch - self.base_pitch)
        yaw_dev = _wrap(yaw - self.base_yaw)
        s.pitch, s.yaw = pitch_dev, yaw_dev

        turned = abs(yaw_dev) > cfg.away_degrees  # eyes look "closed" when turned far

        checks = (
            ("DROWSY_EYES", self.ep_eyes, ear < self.threshold and not turned, cfg.eye_closed_seconds),
            ("YAWN", self.ep_yawn, mar > cfg.mar_threshold, cfg.yawn_seconds),
            ("HEAD_NOD", self.ep_nod, pitch_dev > cfg.nod_degrees, cfg.nod_seconds),
            ("LOOKING_AWAY", self.ep_away, turned, cfg.away_seconds),
        )
        for name, ep, active, min_s in checks:
            duration, fired = ep.update(active, now, min_s)
            if fired:
                self.counts[name] += 1
                self.recent_events.append((now, name))
                self.logger.log(name, duration, ear, mar, pitch_dev, yaw_dev)
                s.new_events.append(name)

        # Fatigue score (time-weighted, so it is independent of camera FPS)
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.5)
        self._last_t = now
        became_high = self._update_fatigue(now, dt, ear < self.threshold and not turned)
        if became_high:
            self.logger.log("FATIGUE_HIGH", 0.0, ear, mar, pitch_dev, yaw_dev)
            s.new_events.append("FATIGUE_HIGH")
            if cfg.sound:
                trigger_alarm()
                self.last_alarm = now
        s.perclos = self.perclos
        s.fatigue_score = self.fatigue_score
        s.fatigue_level = self.fatigue_level

        s.eyes_alert = self.ep_eyes.fired
        s.yawn_alert = self.ep_yawn.fired
        s.nod_alert = self.ep_nod.fired
        s.away_alert = self.ep_away.fired
        s.counts = dict(self.counts)

        # Alarm for safety-critical alerts (eyes closed, head nodding)
        if cfg.sound and (s.eyes_alert or s.nod_alert):
            if now - self.last_alarm >= cfg.alarm_cooldown:
                trigger_alarm()
                self.last_alarm = now

        self._draw_hud(frame, s, (left, right, mouth))
        return frame, s

    # ---- drawing -----------------------------------------------------------
    def _draw_hud(self, frame, s, shapes):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if shapes is not None:
            left, right, mouth = shapes
            if s.eyes_alert:
                eye_color = (0, 0, 255)
            elif self.ep_eyes.since is not None:
                eye_color = (0, 165, 255)
            else:
                eye_color = (0, 255, 0)
            for eye in (left, right):
                cv2.polylines(frame, [eye.astype(np.int32).reshape(-1, 1, 2)],
                              True, eye_color, 1)
            outline = np.array(
                [[mouth_pt[0], mouth_pt[1]] for mouth_pt in _outline_from_mouth(mouth)],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(frame, [outline], True,
                          (0, 0, 255) if s.yawn_alert else (255, 200, 0), 1)

        if s.calibrating:
            if s.face:
                msg = f"Calibrating... keep eyes open, look at camera ({s.calib_progress * 100:.0f}%)"
            else:
                msg = "Calibrating... no face detected"
            cv2.putText(frame, msg, (10, 30), font, 0.6, (0, 255, 255), 2)
        elif not s.face:
            cv2.putText(frame, "No face detected", (10, 30), font, 0.7, (0, 255, 255), 2)

        # Alert banners
        banners = []
        if s.eyes_alert:
            banners.append("DROWSINESS ALERT!")
        if s.nod_alert:
            banners.append("HEAD NODDING!")
        if s.yawn_alert:
            banners.append("YAWNING")
        if s.away_alert:
            banners.append("LOOKING AWAY")
        for i, text in enumerate(banners):
            cv2.putText(frame, text, (10, 70 + 40 * i), font, 1.0, (0, 0, 255), 3)
        if s.eyes_alert or s.nod_alert:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        # Metrics
        if s.face and not s.calibrating:
            cv2.putText(frame, f"EAR {s.ear:.2f} (thr {s.threshold:.2f})   MAR {s.mar:.2f}",
                        (10, h - 65), font, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch {s.pitch:+.0f} deg (down +)   Yaw {s.yaw:+.0f} deg",
                        (10, h - 42), font, 0.55, (255, 255, 255), 2)
        if not s.calibrating:
            col = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}[s.fatigue_level]
            bw = 170
            x0, y0 = w - bw - 10, 10
            cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + 14), (90, 90, 90), 1)
            cv2.rectangle(frame, (x0, y0),
                          (x0 + int(bw * min(s.fatigue_score, 100) / 100), y0 + 14), col, -1)
            cv2.putText(frame, f"Fatigue {s.fatigue_level} {s.fatigue_score:.0f}",
                        (x0, y0 + 36), font, 0.5, col, 2)
            cv2.putText(frame, f"PERCLOS {s.perclos * 100:.0f}%",
                        (x0, y0 + 56), font, 0.5, (255, 255, 255), 2)
        c = s.counts or self.counts
        cv2.putText(frame,
                    f"Eyes:{c.get('DROWSY_EYES', 0)}  Yawns:{c.get('YAWN', 0)}  "
                    f"Nods:{c.get('HEAD_NOD', 0)}  Away:{c.get('LOOKING_AWAY', 0)}",
                    (10, h - 18), font, 0.55, (200, 255, 200), 2)


def _outline_from_mouth(mouth):
    """Re-order the 8 mouth points into a closed outline polygon."""
    up1, lo1, up2, lo2, up3, lo3, left, right = mouth
    return [left, up1, up2, up3, right, lo3, lo2, lo1]