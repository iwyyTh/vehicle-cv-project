import numpy as np
import matplotlib.pyplot as plt


def visualize_image(img: np.darray,
                    title: str = "Image",
                    show_histogram: bool = True,
                    save_path: str = None) -> None:
    """
    Hiển thị ảnh kèm histogram phân bố pixel.
    Args:
        img: numpy array (H, W, 3) hoặc (H,W) - giá trị [0,255] hoặc [0,1]
        title: Tiêu đề hiển thị
        show_histogram: Có vẽ histogram kèm không,
        save_path : Đường dẫn lưu ảnh (None = không lưu)
    """
    # Chuẩn hóa về [0,255] nếu ảnh ở dạng float [0,1]
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    is_gray = img.ndim == 2
    ncols = 2 if show_histogram else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Panel anh
    cmap = "gray" if is_gray else None
    axes[0].imshow(img, cmap=cmap)
    axes[0].set_title(title)
    axes[0].axis("off")

    # Panel histogram
    if show_histogram:
        if is_gray:
            axes[1].hist(img.ravel(), bins=256, range=(
                0, 256), color="gray", alpha=0.7)
        else:
            for i, color in enumerate(["red", "green", "blue"]):
                axes[1].hist(img[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.5,
                             label=color.upper())
            axes[1].legend()
            axes[1].set_title("Histogram")
            axes[1].set_xlabel('Giá trị pixel')
            axes[1].set_ylabel('Số lượng pixel')
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Da luu: {save_path}")
        # plt.show()
        plt.close()
