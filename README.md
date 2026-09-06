# AI-Powered Train Crowd Density Detection and Occupancy Prediction System

Prototype skeleton. Built to grow in stages, not all at once:

1. **Detection** – count people in a coach image using YOLO
2. **Density classification** – map count → Low / Medium / High / Overcrowded
3. **Prediction** – estimate expected crowd for a future train/date using historical patterns
4. **Dashboard** – Streamlit UI tying it all together

## Folder structure

```
train-crowd-density/
├── data/
│   ├── raw/              # original datasets you download (ShanghaiTech, UCF-QNRF, Kaggle sets)
│   ├── processed/        # cleaned/resized data ready for training
│   └── sample_images/    # a few test images for quick manual checks
├── models/                # saved/downloaded model weights (yolov8n.pt etc.) - gitignored
├── src/
│   ├── config.py          # thresholds, paths, constants — single source of truth
│   ├── detection/
│   │   └── detector.py    # YOLO person detection + counting
│   ├── density/
│   │   └── classifier.py  # count -> density level
│   ├── prediction/
│   │   └── predictor.py   # historical-data-based occupancy prediction
│   └── utils/
│       └── helpers.py     # shared small utilities
├── dashboard/
│   └── app.py              # Streamlit dashboard (the "product")
├── scripts/
│   ├── run_detection.py    # CLI: run detection on one image, no dashboard needed
│   └── generate_mock_history.py  # creates fake historical ridership data to develop against
├── tests/
│   └── test_detector.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads YOLOv8's small pretrained weights automatically (needs internet once).

## Run the CLI version first (easiest to debug)

```bash
python scripts/generate_mock_history.py        # creates data/processed/history.csv
python scripts/run_detection.py data/sample_images/sample1.jpg
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Build order (recommended)

1. Get `run_detection.py` working on one sample image — confirms YOLO + OpenCV pipeline works.
2. Tune thresholds in `src/config.py` against a few real coach photos.
3. Wire up `predictor.py` with mock data, confirm dashboard shows a prediction.
4. Swap mock history for a real dataset once the shape/behavior is validated.
5. Only then worry about video streams / real-time — that's a distinct, harder problem from single-image counting.

## Phase 2: entry-point counting + passenger webapp + ticket limiting

Three new pieces, all connected through one backend API:

```
backend/
├── main.py                    # FastAPI: occupancy status, occupancy updates, ticket booking
├── occupancy_store.py         # JSON-file-backed shared state (current count per train)
└── simulate_entry_counter.py  # stands in for a real webcam - reads a video file instead

webapp/
└── index.html                 # passenger-facing mobile-friendly page (plain HTML/JS, no build step)
```

**Why a video file instead of a live webcam:** since there's no physical camera yet, `simulate_entry_counter.py`
reads a video file frame-by-frame instead and pushes counts to the API exactly like a real webcam client would.
Swapping in a real webcam later is a small change — replace `cv2.VideoCapture(video_path)` with
`cv2.VideoCapture(0)` (camera index) in that one file. Everything downstream (API, webapp, booking logic)
doesn't change at all.

**Getting a test video:** you don't need a specialized "crowd counting" dataset for this part, since this
simulates entry counting (people passing a doorway), not overall coach density. Easiest option: film a short
20-30 second clip yourself of people walking through a doorway/hallway and transfer it to your laptop. Save it
to `data/sample_videos/entry.mp4`. Public pedestrian datasets (e.g. Oxford Town Centre, PETS2009) also work if
you'd rather use something citable in your report.

### Running Phase 2

Open three terminals:

```bash
# Terminal 1: the backend API
uvicorn backend.main:app --reload --port 8000

# Terminal 2: the simulated entry camera (adjust the video path/train number)
python backend/simulate_entry_counter.py data/sample_videos/entry.mp4 16526

# Terminal 3: serve the passenger webapp
cd webapp
python -m http.server 5500
# then open http://localhost:5500 in a browser
```

You should see the webapp update automatically every ~5 seconds as the simulator pushes new counts. Try
booking tickets repeatedly on a low-count train (should succeed) versus feeding it a genuinely crowded video
(should eventually get rejected once occupancy crosses the Overcrowded threshold in `src/config.py`).

Note the ticket-limiting rule is currently a hard cutoff (block once Overcrowded) — you may want to explain in
your report why this was chosen over a gradual throttling approach, since that tradeoff is a reasonable thing
an evaluator might ask about.
