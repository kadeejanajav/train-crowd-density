"""
Detects and counts people ("passengers") in a coach image using YOLOv8.

Why YOLO and not a dedicated crowd-counting model (e.g. CSRNet) to start:
YOLO gives bounding boxes, which lets you *see* what it counted (great for
debugging and for demoing to evaluators). Dedicated crowd-counting models
output a density heatmap and are better for genuinely dense crowds (1000s
of people) but are harder to set up and debug. Start with YOLO; swap in a
density-map model later if coach images are too dense for box detection
to work well (e.g. heavily overlapping people).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO

from src.config import YOLO_MODEL_NAME, YOLO_CONFIDENCE_THRESHOLD, PERSON_CLASS_ID


@dataclass
class DetectionResult:
    count: int
    boxes: List[Tuple[float, float, float, float]]  # x1, y1, x2, y2
    confidences: List[float]
    annotated_image_path: str | None = None


class PersonDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        # Loads pretrained COCO weights the first time (downloads automatically).
        self.model = YOLO(model_name)

    def detect(self, image_path: str, save_annotated: bool = True) -> DetectionResult:
        results = self.model.predict(
            source=image_path,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes.xyxy.tolist() if result.boxes is not None else []
        confidences = result.boxes.conf.tolist() if result.boxes is not None else []

        annotated_path = None
        if save_annotated:
            annotated_path = self._save_annotated(result, image_path)

        return DetectionResult(
            count=len(boxes),
            boxes=boxes,
            confidences=confidences,
            annotated_image_path=annotated_path,
        )

    @staticmethod
    def _save_annotated(result, image_path: str) -> str:
        annotated = result.plot()  # numpy array with boxes drawn
        out_path = Path(image_path).with_stem(Path(image_path).stem + "_annotated")
        import cv2
        cv2.imwrite(str(out_path), annotated)
        return str(out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.detection.detector <image_path>")
        sys.exit(1)

    detector = PersonDetector()
    result = detector.detect(sys.argv[1])
    print(f"Detected {result.count} people.")
    print(f"Annotated image saved to: {result.annotated_image_path}")
