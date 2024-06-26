import matplotlib.pyplot as plt

# 새로운 앵커 설정
anchors = {
    'P3/8': [[17,43], [22, 50], [35, 64]],
    'P4/16': [[40, 95], [53, 132], [60, 100]],
    'P5/32': [[75, 148], [100, 232], [180, 250]]
}

# 시각화
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, (key, value) in enumerate(anchors.items()):
    for anchor in value:
        rect = plt.Rectangle((0, 0), anchor[0], anchor[1], fill=False, edgecolor='r', linewidth=2)
        ax[i].add_patch(rect)
    ax[i].set_xlim(0, 640)
    ax[i].set_ylim(0, 512)
    ax[i].set_title(key)
    ax[i].set_aspect('equal')

plt.show()
