"""
Test suite for src/preprocessing/thresholding.py
=================================================
Chạy: python test_thresholding.py
Yêu cầu: opencv-python, numpy

Gồm 2 phần:
  1. Unit Tests  — kiểm tra logic từng class (không cần màn hình)
  2. Visual Test — lưu ảnh so sánh 3 phương pháp ra file PNG
"""

from src.preprocessing.thresholding import (
    SimpleThresholding,
    AdaptiveThresholding,
    OtsuThresholding,
    ThresholdingFactory,
    extract_foreground,
)
import unittest
import numpy as np
import cv2 as cv
import os
import sys

# ── Thêm thư mục gốc project vào sys.path để import được module ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════


def make_gray_image(h: int = 100, w: int = 100, seed: int = 42) -> np.ndarray:
    """Tạo ảnh grayscale giả lặp lại được (dùng cho unit test)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def make_bgr_image(h: int = 100, w: int = 100, seed: int = 42) -> np.ndarray:
    """Tạo ảnh BGR giả (dùng để test extract_foreground)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ════════════════════════════════════════════════════════════
# UNIT TESTS
# ════════════════════════════════════════════════════════════

class TestSimpleThresholding(unittest.TestCase):

    def setUp(self):
        self.img = make_gray_image()
        self.processor = SimpleThresholding(
            thresh_value=127,
            max_value=255,
            thresh_type=cv.THRESH_BINARY,
        )

    def test_output_shape(self):
        """Output phải cùng shape với input."""
        result = self.processor.apply(self.img)
        self.assertEqual(result.shape, self.img.shape)

    def test_output_dtype(self):
        """Output phải là uint8."""
        result = self.processor.apply(self.img)
        self.assertEqual(result.dtype, np.uint8)

    def test_binary_values_only(self):
        """Với THRESH_BINARY, chỉ được có 2 giá trị: 0 và max_value."""
        result = self.processor.apply(self.img)
        unique_vals = set(np.unique(result).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_thresh_binary_inv(self):
        """THRESH_BINARY_INV phải cho kết quả ngược với THRESH_BINARY."""
        normal = SimpleThresholding(127, 255, cv.THRESH_BINARY).apply(self.img)
        inverted = SimpleThresholding(
            127, 255, cv.THRESH_BINARY_INV).apply(self.img)
        # Pixel nào là 255 trong normal thì phải là 0 trong inverted và ngược lại
        np.testing.assert_array_equal(
            normal + inverted, np.full_like(self.img, 255))


class TestAdaptiveThresholding(unittest.TestCase):

    def setUp(self):
        self.img = make_gray_image()
        self.processor = AdaptiveThresholding(
            max_value=255,
            adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type=cv.THRESH_BINARY,
            block_size=11,
            C=2,
        )

    def test_output_shape(self):
        result = self.processor.apply(self.img)
        self.assertEqual(result.shape, self.img.shape)

    def test_output_dtype(self):
        result = self.processor.apply(self.img)
        self.assertEqual(result.dtype, np.uint8)

    def test_binary_values_only(self):
        result = self.processor.apply(self.img)
        unique_vals = set(np.unique(result).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_mean_vs_gaussian(self):
        """ADAPTIVE_THRESH_MEAN_C và GAUSSIAN_C phải cho kết quả khác nhau."""
        mean_result = AdaptiveThresholding(
            255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
        ).apply(self.img)
        gauss_result = self.processor.apply(self.img)
        # Trên ảnh noise ngẫu nhiên, 2 phương pháp thường cho kết quả khác nhau
        self.assertFalse(np.array_equal(mean_result, gauss_result))

    def test_block_size_must_be_odd(self):
        """Block size chẵn phải raise lỗi từ OpenCV."""
        with self.assertRaises(cv.error):
            AdaptiveThresholding(255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, 10, 2).apply(self.img)


class TestOtsuThresholding(unittest.TestCase):

    def setUp(self):
        self.img = make_gray_image()
        self.processor = OtsuThresholding(
            max_value=255,
            thresh_type=cv.THRESH_BINARY,
        )

    def test_output_shape(self):
        result = self.processor.apply(self.img)
        self.assertEqual(result.shape, self.img.shape)

    def test_binary_values_only(self):
        result = self.processor.apply(self.img)
        unique_vals = set(np.unique(result).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_deterministic(self):
        """Cùng một ảnh, Otsu phải cho cùng kết quả."""
        r1 = self.processor.apply(self.img)
        r2 = self.processor.apply(self.img)
        np.testing.assert_array_equal(r1, r2)

    def test_bimodal_image_threshold(self):
        """
        Với ảnh có 2 vùng tối/sáng rõ ràng, Otsu phải tách được hoàn toàn.
        Tạo ảnh: nửa trên = 50 (tối), nửa dưới = 200 (sáng).
        """
        img_bimodal = np.zeros((100, 100), dtype=np.uint8)
        img_bimodal[:50, :] = 50    # vùng tối
        img_bimodal[50:, :] = 200   # vùng sáng
        result = self.processor.apply(img_bimodal)
        # Nửa trên phải là 0, nửa dưới phải là 255
        self.assertTrue(np.all(result[:50, :] == 0))
        self.assertTrue(np.all(result[50:, :] == 255))


class TestThresholdingFactory(unittest.TestCase):

    def test_create_simple(self):
        obj = ThresholdingFactory.create(
            "simple", thresh_value=127, max_value=255, thresh_type=cv.THRESH_BINARY
        )
        self.assertIsInstance(obj, SimpleThresholding)

    def test_create_adaptive(self):
        obj = ThresholdingFactory.create(
            "adaptive",
            max_value=255,
            adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type=cv.THRESH_BINARY,
            block_size=11,
            C=2,
        )
        self.assertIsInstance(obj, AdaptiveThresholding)

    def test_create_otsu(self):
        obj = ThresholdingFactory.create(
            "otsu", max_value=255, thresh_type=cv.THRESH_BINARY
        )
        self.assertIsInstance(obj, OtsuThresholding)

    def test_case_insensitive(self):
        """Factory phải chấp nhận 'Simple', 'SIMPLE', 'simple' như nhau."""
        for name in ["Simple", "SIMPLE", "simple"]:
            obj = ThresholdingFactory.create(
                name, thresh_value=127, max_value=255, thresh_type=cv.THRESH_BINARY
            )
            self.assertIsInstance(obj, SimpleThresholding)

    def test_invalid_type_raises(self):
        """Truyền type không hợp lệ phải raise ValueError."""
        with self.assertRaises(ValueError):
            ThresholdingFactory.create("invalid_method", max_value=255)


class TestExtractForeground(unittest.TestCase):

    def setUp(self):
        self.img_bgr = make_bgr_image()

    def _check_output(self, foreground, mask):
        """Kiểm tra cấu trúc output chung cho mọi method."""
        # foreground phải cùng shape với input BGR
        self.assertEqual(foreground.shape, self.img_bgr.shape)
        # mask phải là ảnh grayscale (2D)
        self.assertEqual(mask.ndim, 2)
        self.assertEqual(mask.shape[:2], self.img_bgr.shape[:2])
        # mask chỉ chứa 0 hoặc 255
        unique_vals = set(np.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_otsu_method(self):
        fg, mask = extract_foreground(self.img_bgr, method="otsu")
        self._check_output(fg, mask)

    def test_simple_method(self):
        fg, mask = extract_foreground(self.img_bgr, method="simple")
        self._check_output(fg, mask)

    def test_adaptive_method(self):
        fg, mask = extract_foreground(self.img_bgr, method="adaptive")
        self._check_output(fg, mask)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            extract_foreground(self.img_bgr, method="unknown")

    def test_foreground_masked_correctly(self):
        """
        Pixel nào mask=0 thì foreground phải là [0,0,0].
        Pixel nào mask=255 thì foreground phải bằng ảnh gốc.
        """
        fg, mask = extract_foreground(self.img_bgr, method="otsu")
        # Vùng bị che
        zero_pixels = (mask == 0)
        np.testing.assert_array_equal(fg[zero_pixels], 0)
        # Vùng giữ nguyên
        keep_pixels = (mask == 255)
        np.testing.assert_array_equal(
            fg[keep_pixels], self.img_bgr[keep_pixels])


# ════════════════════════════════════════════════════════════
# VISUAL TEST — lưu ảnh so sánh 3 phương pháp
# ════════════════════════════════════════════════════════════

def run_visual_test(output_path: str = "results/thresholding_comparison.png"):
    """
    Tạo ảnh lưới so sánh: ảnh gốc + 3 phương pháp threshold.
    Lưu ra file PNG để xem bằng mắt và commit lên GitHub.

    Args:
        output_path: Đường dẫn lưu file ảnh kết quả.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ── Tạo ảnh test có gradient để dễ thấy sự khác biệt ──
    h, w = 200, 300
    gradient = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    # Thêm noise nhẹ để Adaptive có dữ liệu
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 30, size=(h, w), dtype=np.uint8)
    test_img = cv.add(gradient, noise)

    # ── Áp dụng 3 phương pháp ──
    results = {
        "Original": test_img,
        "Simple\n(thresh=127)": SimpleThresholding(127, 255, cv.THRESH_BINARY).apply(test_img),
        "Adaptive\n(Gaussian)":  AdaptiveThresholding(255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv.THRESH_BINARY, 31, 5).apply(test_img),
        "Otsu\n(auto)":           OtsuThresholding(255, cv.THRESH_BINARY).apply(test_img),
    }

    # ── Ghép thành lưới 1×4 ──
    cell_h, cell_w = h + 50, w  # thêm 50px cho label
    canvas = np.ones((cell_h, cell_w * 4 + 30), dtype=np.uint8) * 240

    for i, (title, img) in enumerate(results.items()):
        x = i * (cell_w + 10)
        canvas[40: 40 + h, x: x + cell_w] = img

        # Viết label
        for j, line in enumerate(title.split("\n")):
            cv.putText(canvas, line,
                       (x + 5, 25 + j * 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55,
                       0, 1, cv.LINE_AA)

    cv.imwrite(output_path, canvas)
    print(f"✅ Visual test saved → {output_path}")
    return output_path


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  THRESHOLDING TEST SUITE")
    print("=" * 60)

    # 1. Chạy Unit Tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(
        __file__), pattern="test_thresholding.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 2. Chạy Visual Test
    print("\n" + "=" * 60)
    print("  VISUAL TEST")
    print("=" * 60)
    run_visual_test("results/thresholding_comparison.png")

    # Exit code: 0 nếu pass hết, 1 nếu có lỗi
    sys.exit(0 if result.wasSuccessful() else 1)
