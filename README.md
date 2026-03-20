# 🚗 Vehicle CV Project

Real-time vehicle detection and classification using Computer Vision and Deep Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

## 📌 Overview

This project builds a pipeline to detect and classify vehicles
(cars, motorcycles, trucks, bicycles) from webcam in real-time.

**Stack:** Python · OpenCV · PyTorch · YOLOv8

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/vehicle-cv-project.git
cd vehicle-cv-project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Project Structure

```
vehicle-cv-project/
├── src/
│   ├── preprocessing/   # Image processing modules
│   ├── classical/       # Classical CV (edge, contour, Hough)
│   ├── models/          # CNN & YOLO wrappers
│   └── utils/           # Shared utilities
├── notebooks/           # Jupyter notebooks (weekly)
├── scripts/             # train.py, evaluate.py, demo.py
└── results/             # Metrics, charts, comparisons
```

## 📈 Progress

- [x] Week 1: Python & NumPy Foundation
- [ ] Week 2: OpenCV — Image Preprocessing
- [ ] Week 3: Classical Computer Vision
- [ ] Week 4: Deep Learning with PyTorch
- [ ] Week 5: YOLOv8 & Real-time Demo
