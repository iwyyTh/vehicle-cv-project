import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img_bgr = cv.imread("data/motorbikes/img02.jpg")

if img_bgr is None:
    raise FileNotFoundError("Khong tim thay anh")

print(f"Shape: {img_bgr.shape}")
print(f"Dtype: {img_bgr.dtype}")
print(f"Min max: {img_bgr.min()}/ {img_bgr.max()}")

cv.imshow("BGR img:", img_bgr)
cv.waitKey(0)
cv.destroyAllWindows()

img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
plt.figure(figsize=(8, 5))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()

cv.imwrite("data/raw/processed.jpg", img_bgr)

img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

b, g, r = cv.split(img_bgr)
h, s, v = cv.split(img_hsv)

figure, axes = plt.subplots(2, 4, figsize=(16, 8))

titles = ['BGR (wrong)', 'RGB (correct)', 'Grayscale', 'HSV-H channel',
          'Blue ch.', 'Green ch.', 'Red ch.', 'HSV-S channel']
images = [img_bgr, img_rgb, img_gray, h, b, g, r, s]
cmaps = [None, None, 'gray', 'hsv', 'Blues', 'Greens', 'Reds', 'gray']

for ax, title, im, cmap in zip(axes.flat, titles, images, cmaps):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/color_spaces_comparision.png", dpi=120)
plt.show()

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
mask_white = cv.inRange(img_hsv, lower_white, upper_white)

result = cv.bitwise_and(img_bgr, img_bgr, mask=mask_white)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(mask_white, cmap="gray")
axes[1].set_title('Mask (white regions)')
axes[2].imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
axes[2].set_title('Masked result')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/color_mask_demo.png', dpi=120)
plt.show()
