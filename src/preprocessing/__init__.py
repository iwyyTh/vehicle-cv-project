"""preprocessing — Các module tiền xử lý ảnh cho vehicle detection pipeline.

Subpackage này cung cấp:
- geometry    : Biến đổi hình học (resize, rotate, crop, perspective, affine)
- color_utils : Đọc/ghi ảnh, chuyển đổi color space, tạo color mask

Ví dụ sử dụng nhanh::

    from preprocessing import (
        resize_with_padding, rotate_image, crop_roi,
        load_image, save_image, convert_color, extract_color_mask,
    )

    img    = load_image("frame.jpg", mode="BGR")
    padded = resize_with_padding(img, target_size=(640, 640))
    mask   = extract_color_mask(img, lower, upper, space="HSV")
"""

from .geometry import (
    apply_affine_transform,
    correct_perspective,
    crop_roi,
    resize_with_padding,
    rotate_image,
)
from .color_utils import (
    convert_color,
    display_color_spaces,
    extract_color_mask,
    load_image,
    save_image,
)

__all__ = [
    # geometry
    "resize_with_padding",
    "rotate_image",
    "crop_roi",
    "correct_perspective",
    "apply_affine_transform",
    # color_utils
    "load_image",
    "save_image",
    "convert_color",
    "extract_color_mask",
    "display_color_spaces",
]
