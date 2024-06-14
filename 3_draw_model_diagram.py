# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # 블록을 그리는 함수
# def draw_block(ax, xy, width, height, text, fontsize=8):
#     rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor='black', facecolor='none')
#     ax.add_patch(rect)
#     rx, ry = rect.get_xy()
#     cx = rx + rect.get_width() / 2.0
#     cy = ry + rect.get_height() / 2.0
#     ax.annotate(text, (cx, cy), color='black', weight='bold', 
#                 fontsize=fontsize, ha='center', va='center')

# # 다이어그램 그리기
# fig, ax = plt.subplots(figsize=(12, 20))
# ax.set_xlim(0, 30)
# ax.set_ylim(0, 30)

# # 레이어 추가
# layers = [
#     ("Input (416, 416, 3)", (2, 28), 4, 2),  # Input
#     ("Conv [64, 6, 2, 2]\n(208, 208, 64)", (2, 26), 6, 2),  # Conv Layer 1
#     ("Conv [128, 3, 2]\n(104, 104, 128)", (2, 24), 6, 2),  # Conv Layer 2
#     ("C3 [128, 3]\n(104, 104, 128)", (2, 22), 6, 2),  # C3 Layer 1
#     ("Conv [256, 3, 2]\n(52, 52, 256)", (2, 20), 6, 2),  # Conv Layer 3
#     ("C3 [256, 6]\n(52, 52, 256)", (2, 18), 6, 2),  # C3 Layer 2
#     ("Conv [512, 3, 2]\n(26, 26, 512)", (2, 16), 6, 2),  # Conv Layer 4
#     ("C3 [512, 9]\n(26, 26, 512)", (2, 14), 6, 2),  # C3 Layer 3
#     ("Conv [1024, 3, 2]\n(13, 13, 1024)", (2, 12), 6, 2),  # Conv Layer 5
#     ("C3 [1024, 3]\n(13, 13, 1024)", (2, 10), 6, 2),  # C3 Layer 4
#     ("SPPF [1024, 5]\n(13, 13, 1024)", (2, 8), 6, 2),  # SPPF Layer
#     ("Conv [512, 1, 1]\n(13, 13, 512)", (2, 6), 6, 2),  # Conv Layer 6
#     ("Upsample [None, 2]\n(26, 26, 512)", (2, 4), 6, 2),  # Upsample Layer
#     ("Concat [-1, 6]\n(26, 26, 1024)", (2, 2), 6, 2),  # Concat Layer 1
#     ("C3 [512, 3]\n(26, 26, 512)", (2, 0), 6, 2),  # C3 Layer 5
#     ("Conv [256, 1, 1]\n(26, 26, 256)", (2, -2), 6, 2),  # Conv Layer 7
#     ("Upsample [None, 2]\n(52, 52, 256)", (2, -4), 6, 2),  # Upsample Layer 2
#     ("Concat [-1, 4]\n(52, 52, 512)", (2, -6), 6, 2),  # Concat Layer 2
#     ("C3 [256, 3]\n(52, 52, 256)", (2, -8), 6, 2),  # C3 Layer 6
#     ("Conv [256, 3, 2]\n(26, 26, 256)", (2, -10), 6, 2),  # Conv Layer 8
#     ("Concat [-1, 14]\n(26, 26, 512)", (2, -12), 6, 2),  # Concat Layer 3
#     ("C3 [512, 3]\n(26, 26, 512)", (2, -14), 6, 2),  # C3 Layer 7
#     ("Conv [512, 3, 2]\n(13, 13, 512)", (2, -16), 6, 2),  # Conv Layer 9
#     ("Concat [-1, 10]\n(13, 13, 1024)", (2, -18), 6, 2),  # Concat Layer 4
#     ("C3 [1024, 3]\n(13, 13, 1024)", (2, -20), 6, 2),  # C3 Layer 8
#     ("Detect (P3/8, P4/16, P5/32)", (2, -22), 6, 2)  # Detect Layer
# ]

# for layer in layers:
#     draw_block(ax, layer[1], layer[2], layer[3], layer[0], fontsize=8)

# ax.set_xlim(0, 30)
# ax.set_ylim(0, 30)
# ax.set_aspect('equal')
# plt.gca().invert_yaxis()
# plt.axis('off')
# plt.show()


##########################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 블록을 그리는 함수
def draw_block(ax, xy, width, height, text, color='white'):
    rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width() / 2.0
    cy = ry + rect.get_height() / 2.0
    ax.annotate(text, (cx, cy), color='black', weight='bold', 
                fontsize=10, ha='center', va='center')

# 다이어그램 그리기
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)

# Backbone
draw_block(ax, (0, 8), 2, 1, 'Input\n(416, 416, 3)', color='white')
draw_block(ax, (2, 8), 2, 1, 'Conv [64, 6, 2, 2]\n(208, 208, 64)', color='orange')
draw_block(ax, (4, 8), 2, 1, 'Conv [128, 3, 2]\n(104, 104, 128)', color='orange')
draw_block(ax, (6, 8), 2, 1, 'C3 [128, 3]\n(104, 104, 128)', color='yellow')
draw_block(ax, (8, 8), 2, 1, 'Conv [256, 3, 2]\n(52, 52, 256)', color='orange')
draw_block(ax, (10, 8), 2, 1, 'C3 [256, 6]\n(52, 52, 256)', color='yellow')
draw_block(ax, (12, 8), 2, 1, 'Conv [512, 3, 2]\n(26, 26, 512)', color='orange')
draw_block(ax, (14, 8), 2, 1, 'C3 [512, 9]\n(26, 26, 512)', color='yellow')
draw_block(ax, (16, 8), 2, 1, 'Conv [1024, 3, 2]\n(13, 13, 1024)', color='orange')
draw_block(ax, (18, 8), 2, 1, 'C3 [1024, 3]\n(13, 13, 1024)', color='yellow')
draw_block(ax, (20, 8), 2, 1, 'SPPF [1024, 5]\n(13, 13, 1024)', color='pink')

# Head
draw_block(ax, (18, 6), 2, 1, 'Conv [512, 1, 1]\n(13, 13, 512)', color='white')
draw_block(ax, (16, 6), 2, 1, 'Upsample [None, 2]\n(26, 26, 512)', color='blue')
draw_block(ax, (14, 6), 2, 1, 'Concat [-1, 6]\n(26, 26, 1024)', color='green')
draw_block(ax, (12, 6), 2, 1, 'C3 [512, 3]\n(26, 26, 512)', color='yellow')
draw_block(ax, (10, 6), 2, 1, 'Conv [256, 1, 1]\n(26, 26, 256)', color='white')
draw_block(ax, (8, 6), 2, 1, 'Upsample [None, 2]\n(52, 52, 256)', color='blue')
draw_block(ax, (6, 6), 2, 1, 'Concat [-1, 4]\n(52, 52, 512)', color='green')
draw_block(ax, (4, 6), 2, 1, 'C3 [256, 3]\n(52, 52, 256)', color='yellow')
draw_block(ax, (2, 6), 2, 1, 'Conv [256, 3, 2]\n(26, 26, 256)', color='white')
draw_block(ax, (0, 6), 2, 1, 'Concat [-1, 14]\n(26, 26, 512)', color='green')
draw_block(ax, (-2, 6), 2, 1, 'C3 [512, 3]\n(26, 26, 512)', color='yellow')
draw_block(ax, (-4, 6), 2, 1, 'Conv [512, 3, 2]\n(13, 13, 512)', color='white')
draw_block(ax, (-6, 6), 2, 1, 'Concat [-1, 10]\n(13, 13, 1024)', color='green')
draw_block(ax, (-8, 6), 2, 1, 'C3 [1024, 3]\n(13, 13, 1024)', color='yellow')
draw_block(ax, (-10, 6), 2, 1, 'Detect (P3/8, P4/16, P5/32)', color='white')

plt.gca().invert_yaxis()
plt.axis('off')
plt.show()
