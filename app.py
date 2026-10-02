"""
app.py  -  Streamlit dashboard for the drowsiness detector.

    pip install streamlit pandas
    streamlit run app.py

Runs locally: the webcam and alarm sound are on the machine running Streamlit.
"""

import os
import time
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from drowsiness_core import Config, DrowsinessDetector

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("Real-Time Drowsiness Detection")


# ---- one detector kept across Streamlit reruns --------------------------------
def get_detector(camera_index):
    det = st.session_state.get("detector")
    if det is None or st.session_state.get("camera_index") != camera_index:
        if det is not None:
            det.close()
        cfg = Config(camera_index=camera_index)
        det = DrowsinessDetector(cfg)
        st.session_state["detector"] = det
        st.session_state["camera_index"] = camera_index
    return det


def load_log(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df.iloc[::-1].reset_index(drop=True)  # newest first
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp", "event", "duration_s", "ear", "mar",
                                 "pitch_deg", "yaw_deg"])


def event_chart_data(path):
    """Events per minute, one column per event type (for a stacked bar chart)."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return None
        return (df.groupby([df["timestamp"].dt.floor("min"), "event"])
                  .size().unstack(fill_value=0))
    except Exception:
        return None


# ---- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    run = st.toggle("Start camera", value=False)
    camera_index = int(st.number_input("Camera index", min_value=0, max_value=5, value=0))
    recalibrate = st.button("Recalibrate")

    st.header("Sensitivity")
    sound = st.checkbox("Alarm sound", value=True)
    eye_seconds = st.slider("Eyes closed alert (s)", 0.3, 3.0, 0.7, 0.1)
    ratio = st.slider("Eye threshold ratio", 0.50, 0.95, 0.75, 0.01,
                      help="Alarm EAR = your calibrated open-eye EAR x this ratio")
    mar_thr = st.slider("Yawn MAR threshold", 0.30, 1.00, 0.60, 0.05)
    yawn_seconds = st.slider("Yawn duration (s)", 0.5, 4.0, 1.2, 0.1)
    nod_deg = st.slider("Head nod angle (deg)", 5, 40, 15, 1)
    nod_seconds = st.slider("Head nod duration (s)", 0.3, 3.0, 1.0, 0.1)
    away_deg = st.slider("Looking-away angle (deg)", 20, 60, 35, 1)
    away_seconds = st.slider("Looking-away duration (s)", 0.5, 5.0, 2.0, 0.1)

detector = get_detector(camera_index)

# push slider values into the live config (no recalibration needed)
cfg = detector.cfg
cfg.sound = sound
cfg.eye_closed_seconds = eye_seconds
cfg.threshold_ratio = ratio
cfg.mar_threshold = mar_thr
cfg.yawn_seconds = yawn_seconds
cfg.nod_degrees = float(nod_deg)
cfg.nod_seconds = nod_seconds
cfg.away_degrees = float(away_deg)
cfg.away_seconds = away_seconds

if recalibrate:
    detector.recalibrate()
detector.reset_timers()

with st.sidebar:
    if os.path.exists(cfg.log_path):
        with open(cfg.log_path, "rb") as f:
            st.download_button("Download event log (CSV)", f.read(),
                               file_name="drowsiness_log.csv", mime="text/csv")

# ---- layout -------------------------------------------------------------------
col_video, col_stats = st.columns([3, 2])
with col_video:
    video_slot = st.empty()
with col_stats:
    status_slot = st.empty()
    f1, f2 = st.columns(2)
    fatigue_slot, perclos_slot = f1.empty(), f2.empty()
    fatigue_bar = st.empty()
    m1, m2 = st.columns(2)
    ear_slot, mar_slot = m1.empty(), m2.empty()
    m3, m4 = st.columns(2)
    pitch_slot, yaw_slot = m3.empty(), m4.empty()
    st.subheader("Event counts")
    c1, c2, c3, c4 = st.columns(4)
    eyes_slot, yawn_slot, nod_slot, away_slot = c1.empty(), c2.empty(), c3.empty(), c4.empty()

st.subheader("Event log")
log_slot = st.empty()
log_slot.dataframe(load_log(cfg.log_path))

st.subheader("Charts")
ch1, ch2 = st.columns(2)
with ch1:
    st.caption("Events per minute (from the CSV log)")
    events_chart_slot = st.empty()
with ch2:
    st.caption("Fatigue score during this session (0-100)")
    fatigue_chart_slot = st.empty()

if "fatigue_hist" not in st.session_state:
    st.session_state["fatigue_hist"] = []   # list of (timestamp, score)


def render_event_chart():
    data = event_chart_data(cfg.log_path)
    if data is None:
        events_chart_slot.info("No events logged yet.")
    else:
        events_chart_slot.bar_chart(data)


def render_fatigue_chart():
    hist = st.session_state["fatigue_hist"]
    if len(hist) < 2:
        fatigue_chart_slot.info("Start the camera to record the fatigue score.")
    else:
        df = pd.DataFrame({"Fatigue score": [h[1] for h in hist]},
                          index=[datetime.fromtimestamp(h[0]) for h in hist])  # local time
        fatigue_chart_slot.line_chart(df)


render_event_chart()
render_fatigue_chart()


def update_panel(s):
    if s.calibrating:
        status_slot.info(f"Calibrating... keep eyes open and look at the camera "
                         f"({s.calib_progress * 100:.0f}%)")
    elif not s.face:
        status_slot.warning("No face detected")
    elif s.any_alert:
        names = []
        if s.eyes_alert:
            names.append("Eyes closed")
        if s.nod_alert:
            names.append("Head nodding")
        if s.yawn_alert:
            names.append("Yawning")
        if s.away_alert:
            names.append("Looking away")
        status_slot.error("ALERT: " + ", ".join(names))
    elif s.fatigue_level == "HIGH":
        status_slot.error("HIGH FATIGUE - consider taking a break")
    else:
        status_slot.success("Alert and attentive")

    fatigue_slot.metric("Fatigue level", f"{s.fatigue_level} ({s.fatigue_score:.0f})")
    perclos_slot.metric("PERCLOS (last 60 s)", f"{s.perclos * 100:.0f}%",
                        help="Share of the last minute your eyes were closed. "
                             "Above about 15-20% is a common drowsiness sign.")
    fatigue_bar.progress(int(min(max(s.fatigue_score, 0), 100)))

    ear_slot.metric("EAR", f"{s.ear:.3f}", help=f"Alarm below {s.threshold:.3f}")
    mar_slot.metric("MAR (mouth)", f"{s.mar:.3f}")
    pitch_slot.metric("Pitch (down +)", f"{s.pitch:+.0f} deg")
    yaw_slot.metric("Yaw", f"{s.yaw:+.0f} deg")
    eyes_slot.metric("Eyes", s.counts.get("DROWSY_EYES", 0))
    yawn_slot.metric("Yawns", s.counts.get("YAWN", 0))
    nod_slot.metric("Nods", s.counts.get("HEAD_NOD", 0))
    away_slot.metric("Away", s.counts.get("LOOKING_AWAY", 0))


# ---- main loop ----------------------------------------------------------------
if not run:
    video_slot.info("Turn on **Start camera** in the sidebar to begin.")
else:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        video_slot.error(f"Could not open camera index {camera_index}. "
                         "Close other apps using the camera or try another index.")
    else:
        try:
            n = 0
            last_hist = last_chart = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    video_slot.error("Failed to read from the camera.")
                    break
                annotated, state = detector.process(frame)
                video_slot.image(annotated, channels="BGR")
                if n % 3 == 0:
                    update_panel(state)
                if state.new_events:
                    update_panel(state)
                    log_slot.dataframe(load_log(cfg.log_path))
                    render_event_chart()

                now = time.time()
                if now - last_hist >= 1.0 and not state.calibrating:
                    hist = st.session_state["fatigue_hist"]
                    hist.append((now, state.fatigue_score))
                    del hist[:-900]            # keep the last ~15 minutes
                    last_hist = now
                if now - last_chart >= 3.0:
                    render_fatigue_chart()
                    last_chart = now
                n += 1
        finally:
            cap.release()