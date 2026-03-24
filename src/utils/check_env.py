import importlib
import sys

REQUIRED_PACKAGES = ["numpy", "cv2", "torch", "matplotlib", "PIL", "tqdm"]


def check_environment() -> dict:
    """
    Kiem tra tat ca package bat buoc.
    Returns: dict {'package': 'version' hoac 'MISSING'}
    """
    results = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', 'installed')
            results[pkg] = ver
        except ImportError:
            results[pkg] = 'MISSING'
    return results


if __name__ == '__main__':
    print(f'Python: {sys.version}')
    print('-' * 40)
    for pkg, ver in check_environment().items():
        status = 'OK' if ver != 'MISSING' else 'MISSING !'
        print(f'  {status:8}  {pkg:15} {ver}')
