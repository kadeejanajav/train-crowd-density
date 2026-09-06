"""Create a simple test image with colored rectangles simulating people."""
import cv2
import numpy as np
import os

os.makedirs("data/sample_images", exist_ok=True)

img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray background

# Draw some colored rectangles to simulate a scene
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
for i in range(8):
    x = 50 + (i % 4) * 150
    y = 100 + (i // 4) * 200
    color = colors[i % len(colors)]
    cv2.rectangle(img, (x, y), (x+60, y+150), color, -1)
    cv2.circle(img, (x+30, y-20), 20, (200, 180, 160), -1)  # "head"

cv2.imwrite("data/sample_images/sample1.jpg", img)
print("Test image created: data/sample_images/sample1.jpg")