# SMART PREDICTIVE TRAFFIC INTERSECTION MONITORING USING DEEP LEARNING FOR URBAN TRAFFIC MANAGEMENT SYSTEM

An authentic engineering and academic research project implementing real-time video vehicle detection, multi-object tracking, direction/turning movement analysis, statistical time-series forecasting with Auto-ARIMA, and adaptive traffic signal timing simulation.

---

## 1. Project Objective

Urban road intersections often suffer from severe congestion, inefficient fixed-time signal cycles, and delays caused by asymmetric vehicle queues. This project provides an automated, terminal-based computer-vision and time-series forecasting pipeline that:
1. Takes any user-provided traffic video file (.mp4, .avi, .mov, .mkv).
2. Detects and classifies vehicles into 7 distinct urban categories using a custom-trained **YOLOv8** model (`best.pt`).
3. Tracks individual vehicles across frames and extracts movement trajectories using **BoT-SORT** (`botsort.yaml`).
4. Enforces a configurable **Road Region of Interest (ROI)** to eliminate false detections from trees, sky, buildings, and roadside objects.
5. Accurately counts vehicles by class using unique persistent tracking IDs (eliminating duplicate counts across frames).
6. Detects vehicle turning movements (**STRAIGHT**, **LEFT TURN**, **RIGHT TURN**) by analyzing entry-to-exit zone transitions and trajectory deflection angles.
7. Aggregates observed traffic counts over discrete intervals into a time series and executes **Auto-ARIMA** (`pmdarima.auto_arima`) to forecast short-term traffic volume.
8. Calculates simulated **adaptive green light phase durations** (clamped safely between 20 and 90 seconds) in response to real-time and predicted vehicle demand.
9. Generates a processed output video with visual bounding boxes, tracking IDs, vehicle classes, and a research-grade HUD overlay.
10. Saves detailed logs and structured summaries:
    - `outputs/traffic_counts.csv`
    - `outputs/forecast.csv`
    - `outputs/output.json`
    - `outputs/processed_video.mp4`

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.13** | Core programming language for processing, data structures, and pipeline |
| **OpenCV (`cv2`)** | Video decoding, frame-by-frame manipulation, HUD rendering, and video encoding |
| **Ultralytics YOLOv8** | Deep learning object detection using custom weights (`best.pt`) |
| **BoT-SORT** | Multi-object tracking (`botsort.yaml`) integrating motion and camera compensation |
| **pmdarima (`auto_arima`)** | Automated ARIMA order selection and statistical time-series forecasting |
| **NumPy & Pandas** | High-performance array operations and time-series table logging |
| **Shapely** | Geometric polygon validation for Road ROI filtering and turning zone intersection |

---

## 3. Supported Vehicle Classes

The system utilizes a custom-trained YOLOv8 model (`best.pt`) trained specifically on the following 7 urban vehicle classes:

| Class ID | Class Name | Display Label |
|---|---|---|
| `0` | `car` | Car |
| `1` | `bus` | Bus |
| `2` | `truck` | Truck |
| `3` | `three-wheeler` | Three Wheeler (Auto-rickshaw) |
| `4` | `two-wheeler` | Two Wheeler (Motorcycle / Scooter) |
| `5` | `lcv` | LCV (Light Commercial Vehicle) |
| `6` | `bicycle` | Bicycle |

*(Generic COCO models are not used because COCO lacks separate classes for Three Wheeler and LCV).*

---

## 4. System Architecture & Pipeline

```
┌────────────────────────────────────────────────────────┐
│               User Video Input (.mp4, .avi)            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   OpenCV Video Reader                  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         YOLOv8 Detection + BoT-SORT Tracking           │
│   (BBoxes, Class IDs, Confidence, Persistent IDs)      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              Road ROI Filter (Shapely)                 │
│      Rejects detections outside defined roadway        │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│     Vehicle Counter       │ │  Turning Pattern Detector │
│ Unique track ID binding   │ │ Entry / Exit zone mapping │
│ Class breakdown & totals  │ │ Straight, Left, Right     │
└─────────────┬─────────────┘ └───────────┬───────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│            Time-Series Interval Aggregator             │
│            Observations stored via Pandas              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         Auto-ARIMA Forecasting (pmdarima)              │
│       Predicts upcoming vehicle volume (t + 1)         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Adaptive Green Signal Timing Simulation         │
│     Formula: clamp(Base + Demand * Scale, 20s, 90s)    │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Video HUD Overlay Engine  │ │   File Outputs Exporter   │
│ On-frame text & bboxes    │ │ outputs/processed_video.mp4│
│ Trajectory trails         │ │ outputs/traffic_counts.csv│
│                           │ │ outputs/forecast.csv      │
│                           │ │ outputs/output.json       │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 5. Pipeline Methodologies

### 5.1 Vehicle Detection & False Positive Reduction
- YOLOv8 performs inference on each video frame using `device="cpu"` and resolution (`imgsz=640`).
- A configurable confidence threshold (`CONFIDENCE_THRESHOLD = 0.50`, customizable via `--conf`) filters out low-probability detections.
- Bounding box size thresholds eliminate microscopic background noise artifacts.
- A **Road ROI Polygon** built with Shapely checks whether the lower center of the vehicle's bounding box is inside the active roadway. Detections in background foliage, sky, or buildings are strictly excluded from counts.

### 5.2 BoT-SORT Tracking & Temporal Consistency
- Ultralytics BoT-SORT (`tracker="botsort.yaml"`) maintains track IDs across frames:
  ```python
  model.track(frame, persist=True, tracker="botsort.yaml", conf=0.50, device="cpu")
  ```
- Trajectory history is maintained for each track ID: `(center_x, center_y, frame_number, class_id, confidence)`.
- Temporal consistency requires a vehicle to be tracked for at least `MIN_TRACK_FRAMES = 3` consecutive frames before being confirmed, preventing single-frame detector flicker from registering false counts.

### 5.3 Vehicle Counting
- Every physical vehicle is counted once using its unique tracking ID.
- The counter maintains both **Current Vehicles** (active in current frame) and **Total Vehicles Counted** (cumulative unique vehicles across the session), categorized across all 7 supported classes.

### 5.4 Turning Pattern Detection
- The intersection is divided into configurable entry and exit zones (`NORTH`, `SOUTH`, `EAST`, `WEST`).
- As a vehicle traverses the intersection, its entry zone and exit zone are identified.
- A transition matrix maps zone pairs to turning maneuvers:
  - `(NORTH, SOUTH)` or `(SOUTH, NORTH)` = **Straight**
  - `(NORTH, EAST)` or `(SOUTH, WEST)` = **Left Turn**
  - `(NORTH, WEST)` or `(SOUTH, EAST)` = **Right Turn**
- As a fallback for non-orthogonal camera views, angular displacement deflection across the trajectory points is computed to classify turning angle changes $\Delta \theta$.
- Each vehicle tracking ID is registered at most once for turning movement.

### 5.5 Auto-ARIMA Forecasting
- Vehicle counts are aggregated over discrete time windows (`FORECAST_INTERVAL_SECONDS = 2.0s`).
- While collecting historical data ($N < 5$ intervals), the system reports:
  `"Collecting traffic data... (X/5 intervals)"`
- Once $N \ge 5$ intervals are recorded, `pmdarima.auto_arima()` searches for the optimal $(p, d, q)$ order and produces a 1-step ahead forecast:
  ```python
  model = pm.auto_arima(series, seasonal=False, stepwise=True, max_order=5)
  prediction = model.predict(n_periods=1)
  ```
- The forecast is displayed on the HUD: `AUTO-ARIMA PREDICTED NEXT: XX`.

### 5.6 Adaptive Green Signal Timing
- The forecasted vehicle volume determines the recommended green signal duration:
  $$\text{GreenTime} = \text{clamp}\left(T_{\text{base}} + (\text{Demand} \times S),\ T_{\text{min}},\ T_{\text{max}}\right)$$
  - $T_{\text{base}} = 25\text{s}$ (baseline minimum clearance phase)
  - $\text{Demand} = \text{Forecasted traffic count}$ (or active count during warmup)
  - $S = 2.5\text{s}$ allocated headway per vehicle
  - $T_{\text{min}} = 20\text{s}$ (minimum pedestrian crossing & clearance time)
  - $T_{\text{max}} = 90\text{s}$ (prevents cross-street traffic starvation)
- Displayed on the HUD as: `GREEN LIGHT: XX SECONDS`.

---

## 6. Project Structure

```
traffic-monitoring/
│
├── main.py                    # Primary command-line entry point
├── video_processor.py         # Full video pipeline coordinator & HUD renderer
├── detector.py                # YOLOv8 vehicle detection & ROI filtering
├── tracker.py                 # BoT-SORT multi-object tracking & trajectory management
├── counter.py                 # Unique vehicle counting and class-specific tallies
├── turning_detector.py        # Turning movement detector (Straight, Left, Right)
├── forecasting.py             # Auto-ARIMA time-series model (pmdarima)
├── signal_controller.py       # Simulated adaptive green signal duration calculator
├── config.py                  # System thresholds, classes, and ROI parameters
├── requirements.txt           # Python dependency specification
├── best.pt                    # Custom trained YOLOv8 weights (7 vehicle classes)
│
├── sample_videos/             # Sample traffic video files
│   └── real_traffic.mp4
│
├── outputs/                   # Automatically generated outputs:
│   ├── processed_video.mp4    # Annotated output video
│   ├── traffic_counts.csv     # Frame-by-frame and cumulative vehicle count logs
│   ├── forecast.csv           # Time series interval traffic log
│   └── output.json            # Final summary metrics JSON
│
├── test_model.py              # Standalone test script for inspecting best.pt
├── test_detection.py          # Standalone test script for YOLOv8 + BoT-SORT
├── test_forecasting.py        # Standalone test script for Auto-ARIMA
├── test_pipeline.py           # End-to-end processing test script
└── README.md                  # Complete project documentation
```

---

## 7. Installation & Setup

### 7.1 Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.

### 7.2 Virtual Environment Setup
Open a terminal in the project directory:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 8. Usage

Run the pipeline by passing your video path to `main.py`:

```bash
# Process a traffic video file
python main.py sample_videos/real_traffic.mp4

# Process with a custom confidence threshold (e.g. 0.45)
python main.py sample_videos/real_traffic.mp4 --conf 0.45

# Process a specific number of frames (e.g. first 300 frames)
python main.py sample_videos/real_traffic.mp4 --frames 300

# Display a live OpenCV preview window while processing (press 'q' to stop)
python main.py sample_videos/real_traffic.mp4 --show
```

---

## 9. Final Terminal Report Example

At the conclusion of processing, the system prints a comprehensive report:

```
======================================================================
SMART PREDICTIVE TRAFFIC INTERSECTION MONITORING - FINAL REPORT
======================================================================
Source Video:       sample_videos/real_traffic.mp4
Resolution:         478x850
FPS:                59.94
Frames Processed:   650
Execution Time:     185.4 seconds
----------------------------------------------------------------------
VEHICLE COUNTS (Total Unique Vehicles Counted: 42)
----------------------------------------------------------------------
  Car               : 16
  Bus               : 3
  Truck             : 1
  Three Wheeler     : 5
  Two Wheeler       : 12
  LCV               : 0
  Bicycle           : 5
----------------------------------------------------------------------
TURNING PATTERN COUNTS
----------------------------------------------------------------------
  Straight Count   : 2
  Left Turn Count  : 2
  Right Turn Count : 0
----------------------------------------------------------------------
TRAFFIC FORECAST (Auto-ARIMA)
----------------------------------------------------------------------
  Forecast Status      : AUTO-ARIMA PREDICTED NEXT: 15
  Predicted Next Value : 15 vehicles
  Fitted ARIMA Order   : (1, 0, 0)
  Intervals Collected  : 5
----------------------------------------------------------------------
ADAPTIVE GREEN SIGNAL TIME
----------------------------------------------------------------------
  Green Light Duration : GREEN LIGHT: 62 SECONDS
  Simulation Range     : Min 20s - Max 90s (Base: 25s + 2.5s per vehicle demand)
----------------------------------------------------------------------
OUTPUT FILES
----------------------------------------------------------------------
  Output Video Path    : K:\Projects\traffic-monitoring\outputs\processed_video.mp4
  Traffic Counts CSV   : K:\Projects\traffic-monitoring\outputs\traffic_counts.csv
  Forecast CSV         : K:\Projects\traffic-monitoring\outputs\forecast.csv
  Summary JSON         : K:\Projects\traffic-monitoring\outputs\output.json
======================================================================
```
