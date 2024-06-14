# import os
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from PIL import Image

# def extract_bounding_boxes_from_labels(labels_path):
#     all_boxes = []
#     for filename in os.listdir(labels_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(labels_path, filename)
#             with open(file_path, 'r') as file:
#                 for line in file:
#                     parts = line.strip().split()
#                     _, x_center, y_center, width, height = map(float, parts)
#                     all_boxes.append([width, height])
#     return np.array(all_boxes)

# def calculate_anchors(boxes, n_clusters=9):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(boxes)
#     return kmeans.cluster_centers_, kmeans.labels_

# # 바운딩 박스 추출
# labels_path = '/home/ailab/git/AUE8088-PA2/datasets/nuscenes/train/labels'  # train set의 labels 폴더 경로 설정
# bounding_boxes = extract_bounding_boxes_from_labels(labels_path)
# print(f'Extracted {len(bounding_boxes)} bounding boxes')

# # K-means 클러스터링을 사용하여 앵커 계산
# anchors, labels = calculate_anchors(bounding_boxes, n_clusters=3)
# print(f'Calculated anchors: {anchors}')

# # 이미지 사이즈에 맞는 앵커 계산
# image_path = '/home/ailab/git/AUE8088-PA2/datasets/nuscenes/val/images/0a0d6b8c2e884134a3b48df43d54c36a.png'
# image = Image.open(image_path)
# image_width, image_height = image.size
# print(f'Image size: {image_width}x{image_height}')

# # 정규화된 앵커를 이미지 크기에 맞게 변환
# anchors_scaled = anchors * [image_width, image_height]
# print(f'Scaled anchors: {anchors_scaled}')

# # 정규화된 앵커를 640X640에 맞게 변환
# anchors_scaled_2 = anchors * [640, 640]
# print(f'Scaled anchors (640X640): {anchors_scaled_2}')


# # 앵커 크기 계산 (너비 * 높이)
# anchor_areas = anchors_scaled_2[:, 0] * anchors_scaled_2[:, 1]

# # 앵커를 크기에 따라 정렬
# sorted_indices = np.argsort(anchor_areas)
# sorted_anchors = anchors_scaled_2[sorted_indices]

# # 세 그룹으로 나누기
# anchors_P3 = sorted_anchors[:3]  # 가장 작은 3개
# anchors_P4 = sorted_anchors[3:6] # 중간 3개
# anchors_P5 = sorted_anchors[6:]  # 가장 큰 3개

# # 결과 출력
# print(f'Anchors for P3/8: {anchors_P3}')
# print(f'Anchors for P4/16: {anchors_P4}')
# print(f'Anchors for P5/32: {anchors_P5}')



# # 새 앵커 시각화
# fig, ax = plt.subplots(figsize=(10, 10))
# for anchor in anchors:
#     rect = plt.Rectangle((0, 0), anchor[0] * 640, anchor[1] * 640, fill=False, edgecolor='r', linewidth=2)
#     ax.add_patch(rect)
# ax.set_xlim(0, 640)
# ax.set_ylim(0, 640)
# ax.set_title('Calculated Anchors')
# ax.set_aspect('equal')
# plt.show()

# # 클러스터링 결과 시각화
# fig, ax = plt.subplots(figsize=(10, 10))
# scatter = ax.scatter(bounding_boxes[:, 0] * 640, bounding_boxes[:, 1] * 640, c=labels, cmap='viridis')
# for anchor in anchors:
#     rect = plt.Rectangle((0, 0), anchor[0] * 640, anchor[1] * 640, fill=False, edgecolor='r', linewidth=2)
#     ax.add_patch(rect)
# ax.set_xlim(0, 640)
# ax.set_ylim(0, 640)
# ax.set_title('Clustering of Bounding Boxes')
# ax.set_aspect('equal')
# plt.colorbar(scatter)
# plt.show()


import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

def extract_bounding_boxes_from_labels(labels_path):
    all_boxes = []
    for filename in os.listdir(labels_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    _, x_center, y_center, width, height = map(float, parts)
                    all_boxes.append([width, height])
    return np.array(all_boxes)

def calculate_anchors(boxes, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(boxes)
    return kmeans.cluster_centers_, kmeans.labels_

# 바운딩 박스 추출
labels_path = '/home/ailab/git/AUE8088-PA2/datasets/nuscenes/train/labels'  # train set의 labels 폴더 경로 설정
bounding_boxes = extract_bounding_boxes_from_labels(labels_path)
print(f'Extracted {len(bounding_boxes)} bounding boxes')

# K-means 클러스터링을 사용하여 앵커 계산
anchors, labels = calculate_anchors(bounding_boxes, n_clusters=3)
print(f'Calculated anchors: {anchors}')

# 이미지 사이즈에 맞는 앵커 계산
image_path = '/home/ailab/git/AUE8088-PA2/datasets/nuscenes/val/images/0a0d6b8c2e884134a3b48df43d54c36a.png'
image = Image.open(image_path)
image_width, image_height = image.size
print(f'Image size: {image_width}x{image_height}')

# 정규화된 앵커를 이미지 크기에 맞게 변환
anchors_scaled = anchors * [image_width, image_height]
print(f'Scaled anchors: {anchors_scaled}')

# 정규화된 앵커를 640X640에 맞게 변환
anchors_scaled_2 = anchors * [416, 416]
print(f'Scaled anchors (416X416): {anchors_scaled_2}')

# 앵커 크기 계산 (너비 * 높이)
anchor_areas = anchors_scaled_2[:, 0] * anchors_scaled_2[:, 1]

# 앵커를 크기에 따라 정렬
sorted_indices = np.argsort(anchor_areas)
sorted_anchors = anchors_scaled_2[sorted_indices]

# 세 그룹으로 나누기
anchors_P3 = sorted_anchors[:1]  # 가장 작은 anchor
anchors_P4 = sorted_anchors[1:2] # 중간 anchor
anchors_P5 = sorted_anchors[2:]  # 가장 큰 anchor

# 결과 출력
print(f'Anchors for P3/8: {anchors_P3}')
print(f'Anchors for P4/16: {anchors_P4}')
print(f'Anchors for P5/32: {anchors_P5}')

# 새 앵커 시각화
fig, ax = plt.subplots(figsize=(10, 10))
for anchor in anchors_scaled_2:
    rect = plt.Rectangle((0, 0), anchor[0], anchor[1], fill=False, edgecolor='r', linewidth=2)
    ax.add_patch(rect)
ax.set_xlim(0, 640)
ax.set_ylim(0, 640)
ax.set_title('Calculated Anchors')
ax.set_aspect('equal')
plt.show()

# 클러스터링 결과 시각화
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(bounding_boxes[:, 0] * 640, bounding_boxes[:, 1] * 640, c=labels, cmap='viridis')
for anchor in anchors_scaled_2:
    rect = plt.Rectangle((0, 0), anchor[0], anchor[1], fill=False, edgecolor='r', linewidth=2)
    ax.add_patch(rect)
ax.set_xlim(0, 640)
ax.set_ylim(0, 640)
ax.set_title('Clustering of Bounding Boxes')
ax.set_aspect('equal')
plt.colorbar(scatter)
plt.show()
