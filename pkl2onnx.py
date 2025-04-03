import torch
import torchvision
from l2cs import getArch
import os
import onnx

# 설정
arch = 'ResNet50'      # 사용한 모델 아키텍처
num_bins = 90          # gaze360 기준
pkl_path = 'models/L2CSNet_gaze360.pkl'  # 실제 파일 경로로 바꾸세요
onnx_path = 'output/L2CSNet_gaze360.onnx'

# 파일 존재 여부 확인
if not os.path.exists(pkl_path):
    print(f"오류: '{pkl_path}' 파일이 존재하지 않습니다.")
    exit(1)

# 출력 디렉토리 확인
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

try:
    # 모델 로드
    model = getArch(arch, num_bins)
    model.load_state_dict(torch.load(pkl_path, map_location='cpu'))
    model.eval()
    
    # 더미 입력 (입력 크기는 train.py와 동일하게)
    dummy_input = torch.randn(1, 3, 448, 448)
    
    # ONNX로 export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['yaw', 'pitch'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'yaw': {0: 'batch_size'},
            'pitch': {0: 'batch_size'}
        },
        opset_version=11,
        verbose=False
    )
    
    # ONNX 모델 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX 변환 완료: {onnx_path}")
    print(f"모델 입력 형태: {dummy_input.shape}")
    
except Exception as e:
    print(f"ONNX 변환 중 오류 발생: {e}")