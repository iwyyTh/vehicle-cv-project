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
    target_w, target_h = target_size
    scale = min(target_h/h, target_w/w)
    new_h = int(h*scale)
    new_w = int(w*scale)

    padding_top = (target_h - new_h) // 2
    padding_bot = (target_h - new_h) - padding_top
    padding_right = (target_w - new_w)//2
    padding_left = (target_w - new_w) - padding_right
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


def crop_roi(img: np.ndarray, x: int, y: int,
             w: int, h: int) -> np.ndarray:
    """Cắt vùng ROI từ ảnh.

    Raises:
        ValueError: Nếu tọa độ vượt biên ảnh.
    """
    img_h, img_w = img.shape[:2]

    y_end = y + h
    x_end = x + w

    if (any(k < 0 for k in [x, y])) or (y_end > img_h or x_end > img_w):
        raise ValueError("Toa do vuot bien anh")

    croi = img[y:y_end, x:x_end]
    return croi


def correct_perspective(img: np.ndarray, src_points: np.ndarray,
                        output_size: tuple[int, int] = (400, 300)) -> np.ndarray:
    """Chỉnh perspective từ 4 điểm góc về view phẳng.

    Args:
        src_points: Array shape (4, 2) — 4 góc theo thứ tự
                    [top-left, top-right, bottom-right, bottom-left].
        output_size: (width, height) ảnh output.
    """
    pts_src = np.float32(src_points)
    # tl, tr, br, bl = src_points

    # Tính khoảng cách lớn nhất bằng pytago giúp xác định chiều dài, độ rộng của hình cn sẽ trải phẳng
    # widthA = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
    # widthB = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
    # max_width = max(widthA, widthB)

    # heightA = np.sqrt((abs(tl[0]-bl[0]))**2 + (abs(tl[1]-bl[1]))**2)
    # heightB = np.sqrt((abs(tr[0]-br[0]))**2 + (abs(tr[1]-br[1]))**2)
    # max_height = max(heightA, heightB)

    # dst_points = np.array([
    #     [0, 0],                              # Top-Left
    #     [max_width - 1, 0],                  # Top-Right
    #     [max_width - 1, max_height - 1],     # Bottom-Right
    #     [0, max_height - 1]                  # Bottom-Left
    # ], dtype=np.float32)

    # Lấy width và height trực tiếp từ tham số đầu vào
    dst_w, dst_h = output_size

    # Ép tọa độ đích dàn đều ra đúng bằng kích thước output_size
    dst_points = np.array([
        [0, 0],                      # Top-Left
        [dst_w - 1, 0],              # Top-Right
        [dst_w - 1, dst_h - 1],      # Bottom-Right
        [0, dst_h - 1]               # Bottom-Left
    ], dtype=np.float32)

    # Tạo matran bien doi kich thuoc
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # Nắn chỉnh ảnh
    result = cv2.warpPerspective(img, matrix, output_size)

    return result


def apply_affine_transform(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """Áp dụng affine transform từ 3 cặp điểm tương ứng."""
    h, w = img.shape[:2]
    M = cv2.getAffineTransform(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32),
    )
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


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
roi = crop_roi(img_pad, 0, 0, 300, 300)

src_points_car_side = np.array([
    [70, 90],    # Điểm gần gương chiếu hậu
    [180, 95],   # Điểm gần đuôi xe (trên viền cửa sổ)
    [180, 140],  # Điểm gần bánh sau
    [70, 135]    # Điểm gần bánh trước
], dtype=np.float32)

img_perspective = correct_perspective(img, src_points_car_side)

cv2.imshow("Resize with padding: ", img_pad)
cv2.imshow("Rotate: ", img_pad_rotate)
cv2.imshow("Crop roi: ", roi)
cv2.imshow("Perspective: ", img_perspective)

dst_dir = Path("data/processed")
dst_dir.mkdir(parents=True, exist_ok=True)
dst_dir = dst_dir/"perspective.jpg"

cv2.imwrite(dst_dir, img_perspective)
cv2.waitKey(0)
cv2.destroyAllWindows()
