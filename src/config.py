"""
Configuration constants for the Train Crowd Density system.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Detection settings
YOLO_MODEL = "yolov8n.pt"  # YOLOv8 nano model (auto-downloads on first use)
CONFIDENCE_THRESHOLD = 0.3
PERSON_CLASS_ID = 0  # COCO class ID for 'person'

# Occupancy settings
MAX_CAPACITY = 200  # Maximum passenger capacity of a train carriage

# Density level thresholds (based on occupancy ratio)
DENSITY_THRESHOLDS = {
    "Low": (0.0, 0.3),       # 0% - 30%
    "Medium": (0.3, 0.6),    # 30% - 60%
    "High": (0.6, 0.85),     # 60% - 85%
    "Overcrowded": (0.85, 1.0)  # 85% - 100%
}

# History file
HISTORY_FILE = os.path.join(PROCESSED_DIR, "history.csv")

# Dashboard settings
DASHBOARD_PORT = 8501
