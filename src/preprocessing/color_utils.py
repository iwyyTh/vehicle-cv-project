from __future__ import annotations
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
"""Tiện ích xử lý màu sắc ảnh cho vehicle detection pipeline.

Module này cung cấp wrapper chuẩn hóa trên cv2 để:
- Đọc/ghi ảnh an toàn với error handling
- Chuyển đổi color space linh hoạt
- Tạo binary mask tách vùng màu
"""

# Mapping tên string → hằng số cv2 để hàm convert_color dùng
_SPACE_CODE: dict[tuple[str, str], int] = {
    ('BGR',  'RGB'):  cv2.COLOR_BGR2RGB,
    ('BGR',  'GRAY'): cv2.COLOR_BGR2GRAY,
    ('BGR',  'HSV'):  cv2.COLOR_BGR2HSV,
    ('BGR',  'LAB'):  cv2.COLOR_BGR2LAB,
    ('RGB',  'BGR'):  cv2.COLOR_RGB2BGR,
    ('RGB',  'GRAY'): cv2.COLOR_RGB2GRAY,
    ('RGB',  'HSV'):  cv2.COLOR_RGB2HSV,
    ('HSV',  'BGR'):  cv2.COLOR_HSV2BGR,
    ('GRAY', 'BGR'):  cv2.COLOR_GRAY2BGR,
}


def load_image(path: str, mode="BGR") -> np.ndarray:
    """Đọc ảnh từ file và trả về numpy array theo color space chỉ định.
    ...
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError("Khong tim thay file")

    img = cv2.imread(path_obj)
    if img is None:
        raise ValueError(f"Không thể giải mã hình ảnh tại: {path}")
    key = ("BGR", mode)
    if key in _SPACE_CODE:
        img = cv2.cvtColor(img, _SPACE_CODE[key])
    return img


def save_image(img: np.ndarray, path: str) -> bool:
    """Ghi ảnh ra file, tự tạo thư mục cha nếu chưa có.
    ...
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path_obj), img)
    return success


def convert_color(img: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Chuyển đổi color space linh hoạt bằng tên string.
    ...
    """
    key = (src, dst)
    if key in _SPACE_CODE:
        img = cv2.cvtColor(img, _SPACE_CODE[key])
    return img


def extract_color_mask(
    img: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    space: str = 'HSV',
) -> np.ndarray:
    """
    Args:
        img: Ảnh BGR (uint8) — output trực tiếp từ load_image() hoặc cv2.imread().
            Hàm sẽ tự convert sang `space` trước khi tạo mask.
    """
    key = ("BGR", space)
    if key in _SPACE_CODE:
        convert = cv2.cvtColor(img, _SPACE_CODE[key])
    else:
        convert = img
    return cv2.inRange(convert, lower, upper)


def display_color_spaces(img: np.ndarray) -> None:
    """Hiển thị ảnh gốc và các biểu diễn color space trong một figure.
    ...
    """
    cv2.imshow("Origin: ", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    b, g, r = cv2.split(img)
    h, s, v = cv2.split(img_hsv)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    titles = ['BGR (wrong)', 'RGB (correct)', 'Grayscale', 'HSV-H channel',
              'Blue ch.', 'Green ch.', 'Red ch.', 'HSV-S channel']
    images = [img, img_rgb, img_gray, h, b, g, r, s]
    cmaps = [None, None, 'gray', 'hsv', 'Blues', 'Greens', 'Reds', 'gray']

    for ax, title, im, cmap in zip(axes.flat, titles, images, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()

    save_path = Path('results/color_mask_demo_color_utils.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=120)
    plt.show()


def test_vehicle_pipeline():
    test_dir = Path("data/motorbikes")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_img_path = test_dir / "img02.jpg"

    print("--- Bắt đầu kiểm tra Pipeline ---")

    # 1. Tạo ảnh giả nếu chưa có file thật để test không bị crash[cite: 2]
    # if not test_img_path.exists():
    #     print("! Tạo ảnh test tạm thời...")
    #     dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    #     cv2.rectangle(dummy, (50, 50), (250, 250),
    #                   (0, 255, 0), -1)  # Hình vuông xanh lá
    #     cv2.imwrite(str(test_img_path), dummy)

    try:
        # Test 1: Load ảnh
        img = load_image(str(test_img_path), mode="BGR")
        print(f"[OK] Load ảnh: {img.shape}")

        # Test 2: Convert màu
        img_rgb = convert_color(img, "BGR", "RGB")
        print("[OK] Chuyển đổi BGR -> RGB")

        # Test 3: Tạo Mask (Tìm màu xanh lá)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = extract_color_mask(img, lower_green, upper_green, space='HSV')
        print(f"[OK] Trích xuất Mask (Tìm thấy {np.sum(mask > 0)} pixel)")

        # Test 4: Lưu ảnh đã xử lý
        save_image(mask, "results/mask_result.jpg")
        print("[OK] Đã lưu kết quả vào data/processed/mask_result.jpg")

        # Test 5: Hiển thị
        display_color_spaces(img)

    except Exception as e:
        print(f"[LỖI] {e}")
    finally:
        # Giữ lại file nếu bạn muốn xem, hoặc dùng .unlink() để xóa[cite: 2]
        print("--- Hoàn tất kiểm tra ---")


if __name__ == "__main__":
    test_vehicle_pipeline()
