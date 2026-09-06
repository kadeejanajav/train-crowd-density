"""
Simulates an entry-point webcam. Since there's no physical camera yet,
this reads a VIDEO FILE frame-by-frame instead, runs person detection
every few seconds (not every frame - that would be extremely slow and
is unnecessary), and pushes the count to the backend API - exactly
like a real webcam client would.

Where to get a test video (you don't need a special "crowd counting"
dataset for this part - any footage of people passing through a
doorway/gate works, since this simulates entry counting, not overall
coach crowd estimation):
  - Easiest: film a short 20-30 second clip yourself of people walking
    through a doorway/hallway on your phone, transfer it to your laptop.
  - Or use a public pedestrian dataset with video, e.g. the "Oxford
    Town Centre" dataset or PETS2009 (search these on Kaggle/Google).

Run: python backend/simulate_entry_counter.py data/sample_videos/entry.mp4 16526
"""
import sys
import time

import cv2
import requests

from src.detection.detector import PersonDetector

API_URL = "http://localhost:8000"
SAMPLE_EVERY_N_SECONDS = 3  # how often to run detection - not every frame


def run_simulation(video_path: str, train_number: str):
    detector = PersonDetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(fps * SAMPLE_EVERY_N_SECONDS)
    frame_index = 0

    print(f"Simulating entry camera for train {train_number} using {video_path}")
    print(f"Sampling every {SAMPLE_EVERY_N_SECONDS}s (~every {frame_interval} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            temp_path = "backend/_current_frame.jpg"
            cv2.imwrite(temp_path, frame)

            result = detector.detect(temp_path, save_annotated=False)
            count = result.count

            try:
                resp = requests.post(
                    f"{API_URL}/occupancy/{train_number}/update",
                    json={"count": count},
                    timeout=5,
                )
                resp.raise_for_status()
                print(f"[frame {frame_index}] detected {count} people -> pushed to API")
            except requests.exceptions.RequestException as e:
                print(f"[frame {frame_index}] detected {count} people -> API push FAILED: {e}")
                print("Is the backend running? (uvicorn backend.main:app --reload --port 8000)")

            # Small delay so we're not hammering detection back-to-back
            time.sleep(0.5)

        frame_index += 1

    cap.release()
    print("Video ended.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python backend/simulate_entry_counter.py <video_path> <train_number>")
        sys.exit(1)

    run_simulation(sys.argv[1], sys.argv[2])
