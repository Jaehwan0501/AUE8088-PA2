import os
import json

# 파일 경로 설정
val_file_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/datasets/kaist-rgbt/train-all-04_val_0.2.txt'
images_root_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/datasets/kaist-rgbt/train/images'
labels_root_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/datasets/kaist-rgbt/train/labels'
output_json_path = '/home/ailab/git/AUE8088_MPD/jaehwan/AUE8088-PA2/utils/eval/KAIST_annotation.json'

# 이미지 크기 (고정된 크기)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512

# val 파일 읽기
with open(val_file_path, 'r') as file:
    val_lines = file.readlines()

# 이미지와 주석 데이터를 저장할 리스트
images = []
annotations = []

# 각 이미지와 관련된 정보 및 주석을 추가
for idx, line in enumerate(val_lines):
    image_path = line.strip()
    image_name = os.path.basename(image_path).replace('.jpg', '')
    label_file = os.path.join(labels_root_path, f"{image_name}.txt")

    # 이미지 정보 추가
    images.append({
        "id": idx,
        "im_name": image_path.replace(f"{images_root_path}/", ''),
        "height": IMAGE_HEIGHT,  # 고정된 높이
        "width": IMAGE_WIDTH   # 고정된 너비
    })

    # 주석 파일 읽기
    with open(label_file, 'r') as label:
        lines = label.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            # 정규화된 좌표를 이미지 크기로 변환
            bbox = [
                float(parts[0]) * IMAGE_WIDTH,
                float(parts[1]) * IMAGE_HEIGHT,
                float(parts[2]) * IMAGE_WIDTH,
                float(parts[3]) * IMAGE_HEIGHT
            ]
            height = bbox[3] - bbox[1]
            occlusion = float(parts[4])
            ignore = float(parts[5])

            annotations.append({
                "id": len(annotations),
                "image_id": idx,
                "category_id": 0,  # 'person'으로 고정
                "bbox": bbox,
                "height": height,
                "occlusion": occlusion,
                "ignore": ignore
            })

# JSON 데이터 구성
kaist_annotation = {
    "info": {
        "dataset": "KAIST Multispectral Pedestrian Benchmark",
        "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
        "related_project_url": "http://multispectral.kaist.ac.kr",
        "publish": "CVPR 2015"
    },
    "info_improved": {
        "sanitized_annotation": {
            "publish": "BMVC 2018",
            "url": "https://li-chengyang.github.io/home/MSDS-RCNN/",
            "target": "files in train-all-02.txt (set00-set05)"
        },
        "improved_annotation": {
            "url": "https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn",
            "publish": "BMVC 2016",
            "target": "files in test-all-20.txt (set06-set11)"
        }
    },
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "cyclist"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "person?"}
    ]
}

# JSON 파일로 저장
with open(output_json_path, 'w') as json_file:
    json.dump(kaist_annotation, json_file, indent=4)

print(f"KAIST annotation file created successfully at {output_json_path}")
