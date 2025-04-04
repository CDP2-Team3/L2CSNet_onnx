import cv2
import numpy as np

# BlazeFace Detector using MediaPipe
class BlazeFaceDetector:
    def __init__(self, min_confidence=0.5):
        import mediapipe as mp
        # MediaPipe FaceDetection 모델 초기화 (model_selection 0: Front camera model)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_confidence)
    
    def __call__(self, frame):
        # MediaPipe는 RGB 입력을 사용하므로 변환
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)
        detections = []
        if results.detections:
            for det in results.detections:
                # Bounding box 좌표 (정규화된 좌표를 실제 픽셀 좌표로 변환)
                bboxC = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                xmin = int(bboxC.xmin * w)
                ymin = int(bboxC.ymin * h)
                bw   = int(bboxC.width * w)
                bh   = int(bboxC.height * h)
                xmax = xmin + bw
                ymax = ymin + bh
                box = (xmin, ymin, xmax, ymax)
                score = det.score[0]  # confidence score
                
                # BlazeFace 6 keypoints: 오른쪽눈, 왼쪽눈, 코끝, 입 중심, 오른쪽귀, 왼쪽귀
                keypoints = det.location_data.relative_keypoints
                # 각 키포인트를 픽셀 좌표로 변환
                right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
                left_eye  = (int(keypoints[1].x * w), int(keypoints[1].y * h))
                nose_tip  = (int(keypoints[2].x * w), int(keypoints[2].y * h))
                mouth     = (int(keypoints[3].x * w), int(keypoints[3].y * h))
                # RetinaFace 형식에 맞게 5점 (좌눈, 우눈, 코, 좌입, 우입)으로 변환
                # BlazeFace는 입 중심만 있으므로 좌/우 입꼬리 모두 동일한 입 중심으로 설정
                landmarks = np.array([left_eye, right_eye, nose_tip, mouth, mouth], dtype=int)
                detections.append((box, landmarks, score))
        return detections
