"""
Configuration module for Smart Predictive Traffic Intersection Monitoring System.
Contains all parameters, vehicle classes, thresholds, and region definitions.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
UPLOADS_FOLDER = os.path.join(BASE_DIR, "uploads")

# Ensure directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# Hardware execution
DEVICE = "cpu"
INFERENCE_IMAGE_SIZE = 640

# Custom trained YOLOv8 model classes
VEHICLE_CLASSES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "three-wheeler",
    4: "two-wheeler",
    5: "lcv",
    6: "bicycle"
}

# Formal display labels for HUD and statistics
DISPLAY_NAMES = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Three Wheeler",
    4: "Two Wheeler",
    5: "LCV",
    6: "Bicycle"
}

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
MIN_CONFIDENCE_THRESHOLD = 0.30
MAX_CONFIDENCE_THRESHOLD = 0.70
IOU_THRESHOLD = 0.45

# Tracker settings
TRACKER_CONFIG = "botsort.yaml"
# Minimum number of consecutive detections required before registering count
MIN_TRACK_FRAMES = 3
# Minimum bounding box size (width and height in pixels) to eliminate noise
MIN_BBOX_WIDTH = 15
MIN_BBOX_HEIGHT = 15

# Road Region of Interest (ROI)
# Normalized coordinates [x, y] in range 0.0 to 1.0
# Covers actual intersection roadway and excludes sky, trees, and roadside background
DEFAULT_ROI_NORMALIZED = [
    [0.02, 0.28],
    [0.98, 0.28],
    [0.98, 0.98],
    [0.02, 0.98]
]

# Auto-ARIMA Forecasting Settings
# Observation interval window in seconds for time-series aggregation
FORECAST_INTERVAL_SECONDS = 5.0
# Minimum historical interval observations before running Auto-ARIMA (exactly 2 intervals = 10s video time)
MIN_FORECAST_OBSERVATIONS = 2
# Number of intervals to forecast into the future
FORECAST_STEPS = 1

# Adaptive Green Signal Control Settings (Seconds)
MIN_GREEN_LIGHT = 20
MAX_GREEN_LIGHT = 90
BASE_GREEN_LIGHT = 20
# Additional green seconds per predicted vehicle
SECONDS_PER_VEHICLE = 2.5
# Maximum green time change allowed per cycle (prevents flickering)
MAX_GREEN_CHANGE_PER_CYCLE = 5

# Signal Cycle Settings
# Green time is recalculated only once per cycle (every N frames) to prevent per-frame jumps.
# At 30 fps this is ~3 seconds per cycle; raise to 150 for ~5 s.
SIGNAL_CYCLE_FRAMES = 90
# Exponential Moving Average smoothing factor for demand (0 < alpha <= 1).
# Lower = smoother but slower to react; higher = reacts faster but less stable.
SIGNAL_SMOOTHING_ALPHA = 0.3

# Central Traffic Layout and Road Modes
# Mode: "single" (1 road) or "intersection" (multi-road)
DEFAULT_MODE = "intersection"
DEFAULT_ROAD_COUNT = 4
MODE = DEFAULT_MODE
ROAD_COUNT = DEFAULT_ROAD_COUNT

# Default output file names
OUTPUT_VIDEO_NAME = "processed_video.mp4"
OUTPUT_COUNTS_CSV = "traffic_counts.csv"
OUTPUT_FORECAST_CSV = "forecast.csv"
OUTPUT_SUMMARY_JSON = "summary.json"

# Debug mode (controls display of internal developer logs)
DEBUG = False
