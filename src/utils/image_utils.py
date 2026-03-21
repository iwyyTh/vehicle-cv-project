import numpy as np


def load_image_as_array(filepath: str) -> np.array:
    """
    Đọc ảnh từ file và trả về numpy array

    Args:
        filepath : đường dẫn đến file ảnh

    Returns:
        numpy array shape(H,W,3), dtype float32, giá trị [0,1]
    """
    from PIL import Image
    img = Image.open(filepath).convert("RGB")
    arr = np.array(img, dtype=np.float32)/255.0
    return arr


def print_image_info(img: np.ndarray, ten_anh: str = "Ảnh") -> None:
    """
    In thông tin cơ bản của ảnh ra màn hình.

    Args:
        img: numpy array của ảnh.
        ten_anh: Tên hiển thị.
    """

    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    print(f"{'─'*40}")
    print(f"  {ten_anh}")
    print(f"  Shape : {img.shape}  →  H={h}, W={w}, C={c}")
    print(f"  Dtype : {img.dtype}")
    print(f"  Min   : {img.min():.3f}  |  Max: {img.max():.3f}")
    print(f"  Kích thước: {h}×{w} = {h*w:,} pixels")


if __name__ == "__main__":
    anh_gia = np.random.rand(480, 640, 3).astype(np.float32)
    print_image_info(anh_gia, "Anh test 640x480")
