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
gaze_pipeline = ONNXPipeline(
    onnx_path="./models/L2CSNet_gaze360.onnx",
    device='cpu'
)

# Calibration module
calibrator = Calibrator()

print("Press 'c' to start calibration (9 points). Press 'v' to record each point. Press 'q' to quit.")

#---------------------------------------------------
# Video Capture
#---------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Cannot open webcam.")
    exit(1)

# Create adjustable window of size 800x800
window_name = 'ONNX Gaze Calibration'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 800)

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

    # Mirror for natural interaction
    frame = cv2.flip(frame, 1)

    # Calibration display or gaze rendering
    if calibrator.is_active():
        output_frame = calibrator.display_point(frame)
    else:
        # Gaze estimation
        results = gaze_pipeline.step(frame)
        output_frame = render(frame, results)

        # If calibrated, map gaze to screen coordinates
        if calibrator.is_calibrated():
            pitch = float(results.pitch[0])
            yaw   = float(results.yaw[0])
            sx, sy = calibrator.predict(pitch, yaw)
            cv2.circle(output_frame, (sx, sy), 8, (0, 0, 255), -1)

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
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    # Show output in adjustable window
    cv2.imshow(window_name, output_frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Start calibration
    if key == ord('c') and not calibrator.is_active():
        calibrator.start()
        print('[Calibration] Started. Look at each target and press V.')

    # Record calibration point
    if key == ord('v') and calibrator.is_active():
        # perform gaze estimation to get current pitch/yaw
        gaze = gaze_pipeline.step(frame)
        pitch = float(gaze.pitch[0])
        yaw   = float(gaze.yaw[0])
        calibrator.record(pitch, yaw, frame)
        print(f'[Calibration] Recorded {calibrator.current_index}/9 points.')
        if not calibrator.is_active():
            print('[Calibration] Completed. Calibration model trained.')

# Cleanup
cap.release()
cv2.destroyAllWindows()
