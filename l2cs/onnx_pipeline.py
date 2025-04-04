import onnxruntime as ort
import numpy as np
import cv2
import torch
from .utils import prep_input_numpy
from .results import GazeResultContainer
# from face_detection import RetinaFace
from .detector import BlazeFaceDetector

class ONNXPipeline:
    def __init__(self, onnx_path, device='cpu', include_detector=True, confidence_threshold=0.5):
        self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.include_detector = include_detector
        self.confidence_threshold = confidence_threshold
        self.device = device

        if include_detector:
            # self.detector = RetinaFace()
            self.detector = BlazeFaceDetector(min_confidence=confidence_threshold)
            self.idx_tensor = np.arange(90)
        self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def step(self, frame):
        face_imgs = []
        bboxes, landmarks, scores = [], [], []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None:
                for box, landmark, score in faces:
                    if score < self.confidence_threshold:
                        continue

                    x_min = max(int(box[0]), 0)
                    y_min = max(int(box[1]), 0)
                    x_max = int(box[2])
                    y_max = int(box[3])

                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                pitch, yaw = self.predict_gaze(np.stack(face_imgs))
            else:
                pitch = np.empty((0, 1))
                yaw = np.empty((0, 1))

        else:
            pitch, yaw = self.predict_gaze(frame)

        return GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes),
            landmarks=np.stack(landmarks),
            scores=np.stack(scores)
        )

    def predict_gaze(self, frame):
        if isinstance(frame, np.ndarray):
            img_tensor = prep_input_numpy(frame, torch.device('cpu')).numpy()

        ort_inputs = {self.ort_session.get_inputs()[0].name: img_tensor}
        ort_outs = self.ort_session.run(None, ort_inputs)
        pitch, yaw = ort_outs

        pitch = self.softmax(pitch)
        yaw = self.softmax(yaw)

        pitch_pred = np.sum(pitch * self.idx_tensor, axis=1) * 4 - 180
        yaw_pred = np.sum(yaw * self.idx_tensor, axis=1) * 4 - 180

        pitch_pred = pitch_pred * np.pi / 180.0
        yaw_pred = yaw_pred * np.pi / 180.0

        return pitch_pred, yaw_pred
