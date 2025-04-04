# L2CS-Net Gaze Estimation (Webcam ONNX Version)

L2CS-Net: 기반의 시선 추정 모델을 실시간 웹캠 영상에 적용할 수 있도록 확장한 버전(얼굴인식: mediapipe)

PyTorch `.pkl` 모델을 **ONNX 형식으로 변환**, `onnxruntime`을 활용


## 🔄 주요 변경 사항

| 변경 내용 | 설명 |
|----------|------|
| ✅ `onnx_pipeline.py` | 기존 PyTorch 기반 `Pipeline`을 ONNX 기반 추론이 가능하도록 수정한 `ONNXPipeline` 클래스 추가 |
| ✅ `detector.py` | mediapipe로 얼굴인식을 구현하여, 기존의 retinaface 대체할 수 있도록 `BlazeFaceDetector` 클래스 추가 |
| ✅ `webcamtest_onnx.py` | ONNX 모델을 이용해 실시간 웹캠 시선 추정을 실행할 수 있는 스크립트 추가 |
| ✅ `__init__.py` 수정 | `ONNXPipeline` 클래스, `BlazefaceDetector` 클래스를 외부에서 import 가능하도록 export 처리 |
| ✅ `pkl2onnx.py` | PyTorch `.pkl` 모델을 ONNX 형식으로 변환하는 스크립트  *(선택 사항)* |

---

## 🛠 설치 방법

# Install dependencies
```
pip install -r requirements.txt
```
requirements.txt 설치하고도 안될거임. onnx나 onnxruntime 등 따로 더 설치해야할건데, 그건 쉬울테니 생략

+ 디스코드에 공유한 onnx모델을 models 디렉터리에 넣으면 됨.

---

## 실행
```
python webcamtest_onnx.py
```

---

📄 License
This repository builds upon [TIESLab/L2CS-Net](https://github.com/Ahmednull/L2CS-Net), and follows its license for any base model use. 
