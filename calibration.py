import numpy as np
import cv2
from collections import deque
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class Calibrator:
    """
    Encapsulates the 9-point gaze-to-screen calibration process with optional temporal smoothing.
    Usage:
        calib = Calibrator(smoothing_window=5)
        calib.start()
        # In main loop, while calib.is_active(): frame = calib.display_point(frame)
        # On key 'v': calib.record(pitch, yaw, frame)
        # After calibration: if calib.is_calibrated(): (x, y) = calib.predict(pitch, yaw)
    """
    def __init__(self, smoothing_window=5):
        self.smoothing_window = smoothing_window
        self.pred_buffer = deque(maxlen=smoothing_window)
        self.reset()

    def reset(self):
        # Reset internal state
        self.current_index = 0
        self.gaze_log = []          # list of (pitch, yaw)
        self.screen_points = []     # list of (x, y)
        self.model = None
        self.active = False
        # Clear smoothing buffer
        self.pred_buffer.clear()

    def start(self):
        """Begin calibration session"""
        self.reset()
        self.active = True

    def is_active(self):
        """Return True if calibration session is ongoing"""
        return self.active

    def display_point(self, frame):
        """
        Draw the current calibration target on the input frame.
        Returns annotated frame.
        """
        h, w = frame.shape[:2]
        points = [
            (0, 0), (w // 2, 0), (w - 1, 0),
            (0, h // 2), (w // 2, h // 2), (w - 1, h // 2),
            (0, h - 1), (w // 2, h - 1), (w - 1, h - 1)
        ]
        if self.current_index < len(points):
            pt = points[self.current_index]
            # Draw large green target
            cv2.circle(frame, pt, 15, (0, 255, 0), -1)
            # Draw small red dot at center for precision
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f'Calibration {self.current_index + 1}/9: Look & Press V',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )
        return frame

    def record(self, pitch, yaw, frame):
        """
        Record a gaze sample (pitch, yaw) for the current target.
        After 9 samples, train a Ridge regression model.
        """
        if not self.active:
            return

        h, w = frame.shape[:2]
        points = [
            (0, 0), (w // 2, 0), (w - 1, 0),
            (0, h // 2), (w // 2, h // 2), (w - 1, h // 2),
            (0, h - 1), (w // 2, h - 1), (w - 1, h - 1)
        ]
        self.gaze_log.append((pitch, yaw))
        self.screen_points.append(points[self.current_index])
        self.current_index += 1

        if self.current_index >= len(points):
            self.active = False
            gaze_arr = np.array(self.gaze_log)
            screen_arr = np.array(self.screen_points, dtype=np.float32)
            # Pipeline: standardize inputs then Ridge regression
            self.model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0)
            )
            self.model.fit(gaze_arr, screen_arr)

    def is_calibrated(self):
        """Return True if calibration model has been trained"""
        return self.model is not None

    def predict(self, pitch, yaw):
        """
        Map a new (pitch, yaw) to smoothed screen (x, y).
        Applies moving-average smoothing over recent predictions.
        Raises if model not yet trained.
        """
        if self.model is None:
            raise RuntimeError("Calibrator: model not trained yet")
        raw = self.model.predict(np.array([[pitch, yaw]]))[0]
        x, y = raw[0], raw[1]
        # Append and smooth
        self.pred_buffer.append((x, y))
        xs = [p[0] for p in self.pred_buffer]
        ys = [p[1] for p in self.pred_buffer]
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        return int(round(avg_x)), int(round(avg_y))
