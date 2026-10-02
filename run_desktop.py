"""
run_desktop.py  -  Drowsiness detection in an OpenCV window.

    python run_desktop.py

Keys:  q / ESC = quit     r = recalibrate
Events are logged to drowsiness_log.csv (next to this file).
"""

import sys

import cv2

from drowsiness_core import Config, DrowsinessDetector


def main():
    cfg = Config()  # tweak settings here, e.g. Config(eye_closed_seconds=1.0)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {cfg.camera_index}.")
        sys.exit(1)

    detector = DrowsinessDetector(cfg)
    print(f"Logging events to: {cfg.log_path}")
    print("Keep your eyes open and look at the camera for the first 3 seconds.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            annotated, state = detector.process(frame)
            for event in state.new_events:
                print(f"EVENT: {event}   (fatigue {state.fatigue_level}, score {state.fatigue_score:.0f})")

            cv2.imshow("Drowsiness Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                detector.recalibrate()
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()