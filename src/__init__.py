from src.utils.visualize import visualize_image
from src.utils.image_utils import load_image_as_array
from src.utils.data_loader import DataLoader
from src.utils.file_utils import scan_dataset, get_stats, copy_images
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Đặt TRƯỚC khi import pyplot


loader = DataLoader("data/")
image_paths = loader.scan()

if image_paths:
    img = load_image_as_array(image_paths[0])
    visualize_image(img, title=image_paths[0].name,
                    save_path="results/day3_test.png")
