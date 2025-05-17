import cv2
import time
import numpy as np
import torch

from l2cs import render, ONNXPipeline
from calibration import Calibrator

#---------------------------------------------------
# ONNX Gaze Pipeline and Calibration Setup
#---------------------------------------------------
# Load ONNX-based gaze estimation pipeline
m = ONNXPipeline(
    onnx_path="./models/L2CSNet_gaze360.onnx",
    device='cpu'
)
# Rename for clarity
gaze_pipeline = m

# Calibration module: polynomial regression, smoothing
global_mouse = {'x': 0, 'y': 0}
calibrator = Calibrator(smoothing_window=5, poly_degree=2)

print("Press 'c' to start calibration (9 points). Press 'v' to record points. Press 'q' to quit.")

#---------------------------------------------------
# Video Capture & Window Setup
#---------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Cannot open webcam.")
    exit(1)

window_name = 'ONNX Gaze Calibration'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 800)

# Mouse callback to track position
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global_mouse['x'], global_mouse['y'] = x, y

cv2.setMouseCallback(window_name, mouse_callback)

# FPS variables
prev_time = time.time()
frame_count = 0
fps = 0.0

#---------------------------------------------------
# Main Loop
#---------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Frame read failed.")
        break

    frame = cv2.flip(frame, 1)
    output_frame = frame.copy()

    # Calibration display or gaze rendering
    if calibrator.is_active():
        output_frame = calibrator.display_point(output_frame)
    else:
        results = gaze_pipeline.step(frame)
        output_frame = render(output_frame, results)

        # Gaze prediction and drawing
        if calibrator.is_calibrated():
            p = float(results.pitch[0])
            yv = float(results.yaw[0])
            sx, sy = calibrator.predict(p, yv)
            cv2.circle(output_frame, (sx, sy), 8, (0, 0, 255), -1)
            # Display gaze coordinates
            cv2.putText(
                output_frame,
                f"Gaze: ({sx}, {sy})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        else:
            sx, sy = None, None

    # Draw mouse position and display its coordinates
    mx, my = global_mouse['x'], global_mouse['y']
    cv2.circle(output_frame, (mx, my), 5, (0, 255, 0), -1)
    cv2.putText(
        output_frame,
        f"Mouse: ({mx}, {my})",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

    # FPS computation
    frame_count += 1
    curr_time = time.time()
    elapsed = curr_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        prev_time = curr_time
        frame_count = 0

    cv2.putText(
        output_frame,
        f"FPS: {fps:.2f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )

    # Show frame
    cv2.imshow(window_name, output_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c') and not calibrator.is_active():
        calibrator.start()
        print('[Calibration] Started. Look at each target and press V.')
    if key == ord('v') and calibrator.is_active():
        gaze = gaze_pipeline.step(frame)
        pitch = float(gaze.pitch[0])
        yaw_val = float(gaze.yaw[0])
        calibrator.record(pitch, yaw_val, frame)
        print(f'[Calibration] Recorded {calibrator.current_index}/9 points.')
        if not calibrator.is_active():
            print('[Calibration] Completed. Calibration model trained.')

# Cleanup
cap.release()
cv2.destroyAllWindows()
