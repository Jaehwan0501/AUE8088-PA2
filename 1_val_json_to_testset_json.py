import torch
import os
import json

# 경로 설정
test_file_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/datasets/kaist-rgbt/train-all-04_val_0.2.txt'
images_root_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/datasets/kaist-rgbt/train/images'
output_json_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/runs/test/yolov5n-rgbt2/test_predictions.json'

# YOLOv5 모델 로드 (미리 학습된 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/runs/train/yolov5s-rgbt7/weights/best.pt', force_reload=True)

# test-all-20.txt 파일 읽기
with open(test_file_path, 'r') as file:
    test_lines = file.readlines()

# 예측 결과를 저장할 리스트
predictions = []

# 각 이미지에 대해 예측 수행
for idx, line in enumerate(test_lines):
    image_path = line.strip()
    full_image_path = os.path.join(images_root_path, image_path)
    
    # 이미지 로드 및 예측 수행
    results = model(full_image_path)
    
    # 결과를 JSON 형식으로 변환
    for result in results.xyxy[0]:
        predictions.append({
            "image_id": idx,
            "category_id": int(result[5].item()),  # 예측된 클래스 ID
            "bbox": [float(result[0]), float(result[1]), float(result[2]), float(result[3])],  # Bounding box 좌표
            "score": float(result[4].item())  # 예측된 확률
        })

# JSON 데이터 구성
test_results = {
    "info": {
        "dataset": "KAIST Multispectral Pedestrian Benchmark",
        "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
        "related_project_url": "http://multispectral.kaist.ac.kr",
        "publish": "CVPR 2015"
    },
    "predictions": predictions
}

# JSON 파일로 저장
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w') as json_file:
    json.dump(test_results, json_file, indent=4)

print(f"Test predictions file created successfully at {output_json_path}")
