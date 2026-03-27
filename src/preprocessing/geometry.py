from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

"""Biến đổi hình học cho vehicle detection pipeline.

Tất cả hàm nhận numpy array (H, W, C) uint8 và trả về numpy array.
Tọa độ dùng convention OpenCV: (x, y) = (column, row).
"""


def resize_with_padding(img: np.ndarray, target_size: tuple[int, int] = (640, 640),
                        pad_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Resize ảnh về target_size mà không méo tỷ lệ.

    Args:
        img: Ảnh BGR numpy array shape (H, W, C).
        target_size: (width, height) đích.
        pad_color: Màu padding BGR, mặc định đen.

    Returns:
        Ảnh BGR shape (target_size[1], target_size[0], C).
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_h/h, target_w/w)
    new_h = int(h*scale)
    new_w = int(w*scale)

    padding_top = (new_h - h) // 2
    padding_bot = (new_h - h) // 2
    padding_right = (new_w - w)//2
    padding_left = (new_w - w)//2
    re_img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)

    return cv2.copyMakeBorder(re_img, padding_top, padding_bot, padding_left, padding_right, cv2.BORDER_CONSTANT, value=pad_color)


def rotate_image(img: np.ndarray, angle: float,
                 center: tuple[int, int] | None = None) -> np.ndarray:
    """Xoay ảnh quanh center (mặc định tâm ảnh), giữ nguyên kích thước.

    Args:
        img: Ảnh BGR numpy array.
        angle: Góc xoay (độ), chiều kim đồng hồ dương.
        center: Tâm xoay (x, y). None = tâm ảnh.

    Returns:
        Ảnh đã xoay, cùng shape với ảnh gốc.
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w//2, h//2)
    else:
        center = center
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)

    # rotate_mx[0,0] là cos(theta) và rotate_mx[0,1] là -sin(theta)
    abs_cos = abs(rotate_matrix[0, 0])
    abs_sin = abs(rotate_matrix[0, 1])

    # Tính toán kiến thức mới (bounding box) cho ảnh sau khi xoay
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h*abs_cos + w*abs_sin)
    # Điều chỉnh dịch chuyển trong ma trận
    # Công thức dời tâm cũ sang tâm mới

    rotate_matrix[0, 2] += (new_w / 2) - center[0]
    rotate_matrix[1, 2] += (new_h/2) - center[1]

    rotate_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(new_w, new_h))
    return rotate_img


test_dir = Path("data/cars")
test_dir.mkdir(parents=True, exist_ok=True)
test_dir = test_dir/"img01.jpg"

if not test_dir.exists():
    raise FileNotFoundError("khong tim thay file")

img = cv2.imread(test_dir)

if img is None:
    raise ValueError("khong tim thay anh")
img_pad = resize_with_padding(img)
img_pad_rotate = rotate_image(img_pad, 90)
cv2.imshow("Resize with padding: ", img_pad)
cv2.imshow("Rotate: ", img_pad_rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()
