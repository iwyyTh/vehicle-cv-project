from pathlib import Path
import shutil

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def scan_dataset(folder: str, formats: tuple = SUPPORTED_FORMATS) -> list:
    """
    Quet toan bo anh trong folder (ke ca thu muc con)

    Args:
        folder: Duong dan thu muc can quet
        format: Tuple cac duoi file hop le
    Returns:
        List[Path] : Danh sach duong dan tuyet doi cua file anh tim duoc

    Raises:
        FileNotFound: Neu folder khong ton tai
    """

    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError
    else:
        img_paths = []
        for path in root.rglob('*'):
            if (path.suffix.lower() in formats) and path.is_file():
                img_paths.append(path)

        print(f"Da tim thay {len(img_paths)} anh trong {root}")
    return img_paths


def get_stats(paths: list) -> dict:
    """
    Thong ke so anh theo tung dinh dang

    Args:
        paths: list tra ve tu scan_dataset

    Returns:
        dict: {".jpg" : 5, ...}
    """

    stats = {}
    for path in paths:
        stats[path.suffix.lower()] = stats.get(path.suffix.lower(), 0) + 1
    return stats


def copy_images(src_folder: str, dst_folder: str, formats: tuple = SUPPORTED_FORMATS) -> int:
    """
    Sao chep toan bo anh src den dest

    Args:
        src_folder : thu muc nguon
        dst_folder : thu muc dich
        formats : dinh dang can copy

    Returns:
        int: so file copy thanh cong
    """
    dst = Path(dst_folder)

    dst.mkdir(parents=True, exist_ok=True)
    paths = scan_dataset(src_folder, formats)
    count = 0

    for file in paths:
        file = Path(file)
        try:
            shutil.copy2(file, dst/file.name)
            count = count + 1
        except (FileNotFoundError, PermissionError) as e:
            print(f"Loi khi copy {file.name}: {e}")
            continue
    return count


if __name__ == "__main__":
    images = scan_dataset("data")

    stats = get_stats(images)
    print("Thong ke: ", stats)

    total = copy_images("data", "results/data_flat")
    print(f"Da copy: {total} file")
