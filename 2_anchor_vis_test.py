# import matplotlib.pyplot as plt

# # 앵커 설정
# anchors = {
#     'P3/8': [[10,13], [16,30], [33,23]],
#     'P4/16': [[30,61], [62,45], [59,119]],
#     'P5/32': [[116,90], [156,198], [373,326]]
# }

# # 시각화
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# for i, (key, value) in enumerate(anchors.items()):
#     for anchor in value:
#         rect = plt.Rectangle((0, 0), anchor[0], anchor[1], fill=False, edgecolor='r', linewidth=2)
#         ax[i].add_patch(rect)
#     ax[i].set_xlim(0, 400)
#     ax[i].set_ylim(0, 400)
#     ax[i].set_title(key)
#     ax[i].set_aspect('equal')

# plt.show()

import matplotlib.pyplot as plt

# 새로운 앵커 설정
anchors = {
    'P3/8': [[16, 30]],
    'P4/16': [[62, 45]],
    'P5/32': [[156, 198]]
}

# 시각화
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, (key, value) in enumerate(anchors.items()):
    for anchor in value:
        rect = plt.Rectangle((0, 0), anchor[0], anchor[1], fill=False, edgecolor='r', linewidth=2)
        ax[i].add_patch(rect)
    ax[i].set_xlim(0, 400)
    ax[i].set_ylim(0, 400)
    ax[i].set_title(key)
    ax[i].set_aspect('equal')

plt.show()
