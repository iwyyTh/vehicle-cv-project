import sys
from pathlib import Path

# Thêm src/ vào sys.path để pytest tìm thấy package preprocessing
sys.path.insert(0, str(Path(__file__).parent / "src"))
