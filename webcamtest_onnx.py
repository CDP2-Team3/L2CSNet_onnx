from l2cs import render
from l2cs import ONNXPipeline # 새로 만든 ONNXPipeline
import cv2
import time

# ONNX 기반 파이프라인 로딩, 새로 만든 ONNXPipeline 사용
gaze_pipeline = ONNXPipeline(
    onnx_path="./models/L2CSNet_gaze360.onnx",
    device='cpu'
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

prev_time = time.time()
frame_count = 0
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        start_time = time.time()

        results = gaze_pipeline.step(frame)
        output_frame = render(frame, results)

        frame_count += 1
        elapsed_time = time.time() - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = time.time()
            frame_count = 0

        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ONNX Gaze Estimation - Live', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
