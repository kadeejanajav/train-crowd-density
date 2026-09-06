"""
Person detection module using YOLOv8.
Detects people in images and calculates crowd density metrics.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, PERSON_CLASS_ID,
    MAX_CAPACITY, DENSITY_THRESHOLDS
)


class CrowdDetector:
    """Detects people in images using YOLOv8 and computes occupancy metrics."""

    def __init__(self, model_path=None, max_capacity=None):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model weights. Defaults to config value.
            max_capacity: Max passenger capacity. Defaults to config value.
        """
        self.model_path = model_path or YOLO_MODEL
        self.max_capacity = max_capacity or MAX_CAPACITY
        self.model = YOLO(self.model_path)

    def detect_people(self, image_input):
        """
        Detect people in an image.
        """

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("image_input must be a file path (str) or numpy array")

        results = self.model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

        boxes = []
        confidences = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id == PERSON_CLASS_ID:
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])

                    boxes.append(coords)
                    confidences.append(conf)

        return {
            "count": len(boxes),
            "boxes": boxes,
            "confidences": confidences,
            "results": results
        }

    def compute_occupancy(self, people_count):
        """
        Compute occupancy ratio and density level.
        Applies an occlusion correction because YOLO
        cannot detect every passenger in a crowded coach.
        """

        # Estimate hidden passengers
        if people_count <= 5:
            estimated_people = people_count
        elif people_count <= 10:
            estimated_people = int(people_count * 1.5)
        elif people_count <= 20:
            estimated_people = int(people_count * 1.8)
        else:
            estimated_people = int(people_count * 2.0)

        estimated_people = min(estimated_people, self.max_capacity)

        ratio = estimated_people / self.max_capacity

        level = "Unknown"

        for label, (low, high) in DENSITY_THRESHOLDS.items():
            if low <= ratio < high:
                level = label
                break

        if ratio >= 1.0:
            level = "Overcrowded"

        return {
            "count": estimated_people,
            "detected_count": people_count,
            "occupancy_ratio": round(ratio, 4),
            "occupancy_percent": round(ratio * 100, 2),
            "density_level": level,
            "max_capacity": self.max_capacity
        }

    def annotate_image(self, image_input, detection_result):
        """
        Draw bounding boxes and labels on the image.
        """

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()

        for box, conf in zip(
                detection_result["boxes"],
                detection_result["confidences"]):

            x1, y1, x2, y2 = [int(c) for c in box]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"Person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return image

    def process_image(self, image_input):
        """
        Full pipeline.
        """

        detection = self.detect_people(image_input)

        occupancy = self.compute_occupancy(
            detection["count"]
        )

        annotated = self.annotate_image(
            image_input,
            detection
        )

        return {
            "detection": detection,
            "occupancy": occupancy,
            "annotated_image": annotated
        }