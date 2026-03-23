import pytest
from pathlib import Path
from src.utils.file_utils import scan_dataset, get_stats, copy_images


@pytest.fixture
def fake_dataset(tmp_path):
    """Tạo bộ ảnh giả trong thư mục tạm."""
    (tmp_path / 'cats').mkdir()
    (tmp_path / 'dogs').mkdir()
    # Ảnh hợp lệ
    for name in ['a.jpg', 'b.PNG', 'c.jpeg']:
        (tmp_path / 'cats' / name).touch()
    for name in ['d.png', 'e.bmp']:
        (tmp_path / 'dogs' / name).touch()
    # File không phải ảnh — không được đếm
    (tmp_path / 'readme.txt').touch()
    (tmp_path / 'dogs' / '.DS_Store').touch()
    return tmp_path


def test_scan_finds_all_images(fake_dataset):
    result = scan_dataset(str(fake_dataset))
    assert len(result) == 5        # 3 cats + 2 dogs


def test_scan_ignores_non_images(fake_dataset):
    result = scan_dataset(str(fake_dataset))
    exts = [p.suffix.lower() for p in result]
    assert '.txt' not in exts


def test_scan_case_insensitive(fake_dataset):
    result = scan_dataset(str(fake_dataset))
    # b.PNG (uppercase) phải được tìm thấy
    names = [p.name.lower() for p in result]
    assert 'b.png' in names


def test_scan_raises_if_folder_not_found():
    with pytest.raises(FileNotFoundError):
        scan_dataset('/path/that/does/not/exist')


def test_get_stats_counts_correctly(fake_dataset):
    paths = scan_dataset(str(fake_dataset))
    stats = get_stats(paths)
    # .jpg + .jpeg = 2, .png (+ .PNG normalized) = 2, .bmp = 1
    assert stats.get('.jpg', 0) + stats.get('.jpeg', 0) == 2
    assert stats.get('.png', 0) == 2
    assert stats.get('.bmp', 0) == 1


def test_copy_images_creates_flat_structure(fake_dataset, tmp_path):
    dst = tmp_path / 'flat_out'
    count = copy_images(str(fake_dataset), str(dst))
    assert count == 5
    # Tất cả ảnh phải nằm thẳng trong dst (không cấu trúc con)
    copied = list(dst.glob('*'))
    assert len(copied) == 5
