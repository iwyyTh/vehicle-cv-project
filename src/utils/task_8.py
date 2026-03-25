import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img_bgr = cv.imread("data/cars/img01.jpg")

if img_bgr is None:
    raise FileNotFoundError("Khong tim thay anh")

print(f"Shape: {img_bgr.shape}")
print(f"Dtype: {img_bgr.dtype}")
print(f"Min max: {img_bgr.min()}/ {img_bgr.max()}")

cv.imshow("BGR img:", img_bgr)
cv.waitKey(0)
cv.destroyAllWindows()

img_bgr = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
plt.figure(figsize=(8, 5))
plt.imshow(img_bgr)
plt.axis("off")
plt.show()

cv.imwrite("data/raw/processed.jpg", img_bgr)
