import os
from pathlib import Path


class DataLoader:
    """
    Class quản lý việc tải danh sách ảnh từ một thư mục.

    Attributes:
        data_dir (Path): Đường dẫn tới thư mục chứa ảnh.
        supported_formats (tuple): Các định dạng ảnh được chấp nhận.
    """

    # Định dạng ảnh hỗ trợ — dùng CONSTANT (viết hoa) cho giá trị không đổi
    SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def __init__(self, data_dir: str):
        """
        Khởi tạo DataLoader.

        Args:
            data_dir: Đường dẫn tới thư mục chứa ảnh.

        Raises:
            FileNotFoundError: Nếu thư mục không tồn tại.
        """
        self.data_dir = Path(data_dir)

        # Kiểm tra thư mục có tồn tại không
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {self.data_dir}")

        print(f"✅ DataLoader khởi tạo thành công: {self.data_dir}")

    def scan(self) -> list:
        """
        Quét toàn bộ ảnh trong thư mục (kể cả thư mục con).

        Returns:
            Danh sách đường dẫn tuyệt đối của các file ảnh.
        """
        image_paths = []

        for path in self.data_dir.rglob("*"):          # rglob quét đệ quy
            if path.suffix.lower() in self.SUPPORTED_FORMATS:
                image_paths.append(path)

        print(f"📁 Tìm thấy {len(image_paths)} ảnh trong '{self.data_dir}'")
        return image_paths

    def summary(self) -> dict:
        """
        Thống kê số lượng ảnh theo từng định dạng.

        Returns:
            Dict dạng {'.jpg': 10, '.png': 5, ...}
        """
        images = self.scan()
        stats = {}

        for path in images:
            ext = path.suffix.lower()
            # Tăng đếm, mặc định 0 nếu chưa có
            stats[ext] = stats.get(ext, 0) + 1

        return stats


# ── Chạy thử khi gọi trực tiếp file này ──────────────────────────
if __name__ == "__main__":
    # Test với thư mục data/ (tạo vài ảnh giả để thử)
    loader = DataLoader("data")
    stats = loader.summary()
    print("📊 Thống kê:", stats)
