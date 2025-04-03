# import cv2
# import numpy as np
# import onnxruntime as ort
# import time
# from l2cs import draw_gaze
# from face_detection import RetinaFace
# from l2cs.utils import transformations

# # 경로 설정
# onnx_model_path = "./models/L2CSNet_gaze360.onnx"  # 실제 변환된 ONNX 모델 경로로 수정하세요

# def softmax(x):
#     x_exp = np.exp(x - np.max(x))
#     return x_exp / x_exp.sum()

# # ONNX Runtime 세션 생성
# session = ort.InferenceSession(onnx_model_path)
# input_name = session.get_inputs()[0].name
# output_names = [o.name for o in session.get_outputs()]

# # RetinaFace 얼굴 검출기
# detector = RetinaFace()

# # gaze 예측 시 사용할 인덱스 텐서
# idx_tensor = np.arange(90)

# # 웹캠 캡처
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("웹캠을 열 수 없습니다.")
#     exit()

# prev_time = time.time()
# frame_count = 0
# fps = 0

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = detector(frame_rgb)

#         if faces is not None:
#             for box, landmark, score in faces:
#                 if score < 0.5:
#                     continue

#                 x_min = max(int(box[0]), 0)
#                 y_min = max(int(box[1]), 0)
#                 x_max = int(box[2])
#                 y_max = int(box[3])

#                 face = frame_rgb[y_min:y_max, x_min:x_max]
#                 face = cv2.resize(face, (448, 448))
#                 face_input = transformations(face).unsqueeze(0).numpy()

#                 # ONNX 추론
#                 yaw_out, pitch_out = session.run(output_names, {input_name: face_input})

#                 # Softmax 후 연속 각도로 변환
#                 pitch_pred = np.sum(softmax(pitch_out[0]) * idx_tensor) * 4 - 180
#                 yaw_pred = np.sum(softmax(yaw_out[0]) * idx_tensor) * 4 - 180

#                 # 라디안으로 변환
#                 pitch_rad = pitch_pred * np.pi / 180.0
#                 yaw_rad = yaw_pred * np.pi / 180.0

#                 # 시선 시각화
#                 draw_gaze(x_min, y_min, x_max - x_min, y_max - y_min, frame, (pitch_rad, yaw_rad), color=(0, 255, 0))

#         # FPS 계산
#         frame_count += 1
#         elapsed_time = time.time() - prev_time
#         if elapsed_time >= 1.0:
#             fps = frame_count / elapsed_time
#             prev_time = time.time()
#             frame_count = 0

#         cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow('ONNX Gaze Estimation', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     cap.release()
#     cv2.destroyAllWindows()


# def softmax(x):
#     x_exp = np.exp(x - np.max(x))
#     return x_exp / x_exp.sum()

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
