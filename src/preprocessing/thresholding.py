from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
import pathlib as Path


class BaseThresholding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, Img: np.ndarray) -> np.ndarray:
        pass


class SimpleThresholding(BaseThresholding):
    def __init__(self, thresh_value: int, max_value: int, thresh_type: str) -> None:
        super().__init__()
        self.thresh_value = thresh_value
        self.max_value = max_value
        self.thresh_type = thresh_type

    def apply(self, Img):
        return cv.threshold(
            Img,
            self.thresh_value,
            self.max_value,
            self.thresh_type)[1]


class AdaptiveThresholding(BaseThresholding):
    def __init__(self, max_value: int, adaptive_method: int, thresh_type: int, block_size: int, C: float) -> None:
        super().__init__()
        self.max_value = max_value
        self.adaptive_method = adaptive_method
        self.thresh_type = thresh_type
        self.block_size = block_size
        self.C = C

    def apply(self, Img):
        # Lưu ý: adaptiveThreshold chỉ trả về 1 giá trị là ảnh (không có ret như threshold thường)
        return cv.adaptiveThreshold(
            Img,
            self.max_value,
            self.adaptive_method,
            self.thresh_type,
            self.block_size,
            self.C
        )


class OtsuThresholding(BaseThresholding):
    def __init__(self, max_value: int, thresh_type: int) -> None:
        super().__init__()
        self.max_value = max_value
        self.thresh_type = thresh_type

    def apply(self, Img):
        # Truyền 0 vào vị trí thresh_value vì thuật toán Otsu sẽ tự động tính toán ngưỡng tối ưu
        # Phải cộng (hoặc dùng toán tử |) cờ cv.THRESH_OTSU vào thresh_type
        return cv.threshold(
            Img,
            0,
            self.max_value,
            self.thresh_type | cv.THRESH_OTSU
        )[1]


class ThresholdingFactory:
    # Một dictionary dùng để ánh xạ (map) chuỗi string với đúng Class tương ứng
    _methods = {
        "simple": SimpleThresholding,
        "adaptive": AdaptiveThresholding,
        "otsu": OtsuThresholding
    }

    @classmethod
    def create(cls, threshold_type: str, **kwargs) -> BaseThresholding:
        """
        Khởi tạo object Thresholding dựa trên type truyền vào.
        **kwargs sẽ gom toàn bộ các tham số động để truyền vào __init__ của class con.
        """
        # Chuyển string về chữ thường để tránh lỗi gõ hoa/thường (ví dụ "Simple" hay "simple" đều nhận)
        method_class = cls._methods.get(threshold_type.lower())

        if method_class is None:
            raise ValueError(
                f"Loại threshold '{threshold_type}' không tồn tại! Hãy chọn: {list(cls._methods.keys())}")

        # method_class(**kwargs) tương đương với việc gọi SimpleThresholding(thresh_value=127, ...)
        return method_class(**kwargs)


def extract_foreground(img: np.ndarray, method="otsu") -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo foreground và mask cho img truyền vào với mask được làm theo method

    Args:
        img : mảnh numpy của hình ảnh
        method : phương thức tạo mask

    Returns:
        tuple(foreground, mask)

    """

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    processor = None

    if method.lower() == "simple":
        processor = ThresholdingFactory.create(
            threshold_type="simple",
            thresh_value=127,
            max_value=255,
            thresh_type=cv.THRESH_BINARY
        )
    elif method.lower() == "adaptive":
        processor = ThresholdingFactory.create(
            threshold_type="adaptive",
            max_value=255,
            adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type=cv.THRESH_BINARY,
            block_size=11,
            C=22
        )
    elif method.lower() == "otsu":
        processor = ThresholdingFactory.create(
            threshold_type="otsu",
            max_value=255,
            thresh_type=cv.THRESH_BINARY
        )
    else:
        raise ValueError("Not type can support")

    mask = processor.apply(img_gray)
    and_img = cv.bitwise_and(img, img, mask=mask)
    return (and_img, mask)
