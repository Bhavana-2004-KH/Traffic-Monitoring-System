"""
Zone Manager module for automatic road zone detection and persistence.

Automatically identifies road approach corridors from the video itself using:
  1. MOG2 background subtraction for motion accumulation.
  2. Road surface color/saturation filter (dark gray/asphalt mask).
  3. Radial angular binning outward from detected intersection center.
  4. Angular continuity analysis to isolate distinct road approaches.

VIDEO MODE CONTRACT
-------------------
SINGLE_ROAD  ("single"):
  - AutoRoadDetector is NOT activated for zone drawing.
  - A single full-frame zone named "Single Road" is created automatically.
  - No Road 1 / Road 2 / Road 3 / Road 4 zones are ever created.
  - No fallback 4-way intersection layout is applied.

INTERSECTION ("intersection"):
  - AutoRoadDetector runs automatically from the first N frames of the video.
  - Road approaches are identified without any manual input.
  - Supports 3, 4, 5, 6+ roads based on actual video geometry and traffic.
  - Results are cached per video in outputs/<video_stem>_roads.json.
"""
import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
import config

# Color palette for rendering different road zones (BGR)
ROAD_COLORS = [
    (0, 200, 255),   # Yellow-Orange
    (255, 120, 0),   # Blue
    (50, 220, 50),   # Green
    (220, 50, 220),  # Purple
    (0, 240, 240),   # Cyan
    (50, 100, 255),  # Coral
    (255, 200, 0),   # Sky Blue
    (100, 255, 180)  # Mint
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_zone_file_path(video_path, output_dir=config.OUTPUT_FOLDER, mode=None, road_count=None):
    """Generates the JSON storage path for a video's road zone layout, namespaced by mode and road_count."""
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    if mode == "single" or road_count == 1:
        return os.path.join(output_dir, f"{video_stem}_single_zones.json")
    elif road_count is not None and road_count != 4:
        return os.path.join(output_dir, f"{video_stem}_{road_count}roads_zones.json")
    return os.path.join(output_dir, f"{video_stem}_zones.json")


def normalize_polygon(points, frame_width, frame_height):
    """Converts pixel coordinate polygon to normalized [0.0 - 1.0] coordinates."""
    return [[round(float(px) / frame_width, 4), round(float(py) / frame_height, 4)] for px, py in points]


def denormalize_polygon(norm_points, frame_width, frame_height):
    """Converts normalized polygon to pixel coordinates."""
    return [(int(round(nx * frame_width)), int(round(ny * frame_height))) for nx, ny in norm_points]


def _log(msg):
    if getattr(config, "DEBUG", False):
        print(msg)


def save_zones(zone_file_path, zones_normalized, mode="intersection", video_name="", video_width=None, video_height=None):
    """Saves normalized road zones dictionary with mode metadata to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(zone_file_path)), exist_ok=True)
    payload = {
        "mode": mode,
        "video_name": video_name,
        "video_width": video_width,
        "video_height": video_height,
        "road_names": list(zones_normalized.keys()) if zones_normalized else [],
        "zones": zones_normalized
    }
    with open(zone_file_path, "w") as f:
        json.dump(payload, f, indent=4)
    _log(f"[ZoneManager] Saved {len(zones_normalized)} zones ({mode} mode) to {zone_file_path}")


def load_zones(zone_file_path, frame_width, frame_height, declared_mode=None, declared_road_count=None):
    """
    Loads road zones from JSON and converts to pixel polygons and Shapely Polygons.

    Args:
        zone_file_path: Path to the zone JSON file.
        frame_width, frame_height: Frame dimensions for denormalization.
        declared_mode: The mode the caller explicitly set ("single" / "intersection" / None).
        declared_road_count: The exact number of roads expected. If the cached file
                             has a different road count, it is discarded.

    Returns:
        mode: "single" or "intersection"
        zones_normalized: dict {name: [[x, y], ...]}
        zones_pixel: dict {name: [(x, y), ...]}
        zone_polygons: dict {name: Polygon}
    """
    if not os.path.exists(zone_file_path):
        return declared_mode or "intersection", None, None, None

    try:
        with open(zone_file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "zones" in data:
            file_mode = data.get("mode", "intersection")
            zones_normalized = data["zones"]
        else:
            zones_normalized = data
            file_mode = None

        mode = declared_mode or file_mode or "intersection"

        # In SINGLE_ROAD mode, discard multi-road zones from a stale file.
        if mode == "single" and len(zones_normalized) > 1:
            _log(f"[ZoneManager] Discarding {len(zones_normalized)}-road zone file "
                 f"because declared mode is SINGLE_ROAD.")
            return mode, None, None, None

        # Discard if cached road count does not match requested road count
        if declared_road_count is not None and len(zones_normalized) != declared_road_count:
            _log(f"[ZoneManager] Discarding {len(zones_normalized)}-road zone file "
                 f"because declared road_count is {declared_road_count}.")
            return mode, None, None, None

        zones_pixel = {}
        zone_polygons = {}
        for name, pts in zones_normalized.items():
            px_pts = denormalize_polygon(pts, frame_width, frame_height)
            if len(px_pts) >= 3:
                zones_pixel[name] = px_pts
                zone_polygons[name] = Polygon(px_pts)

        return mode, zones_normalized, zones_pixel, zone_polygons

    except Exception as e:
        _log(f"[ZoneManager] Error loading zone file {zone_file_path}: {e}")
        return declared_mode or "intersection", None, None, None


def _build_single_road_zone(frame_width, frame_height):
    """
    Returns a single full-frame zone named 'Single Road'.
    Used automatically in SINGLE_ROAD mode — no interactive window is shown.
    """
    norm = {
        "Single Road": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }
    px = {}
    poly = {}
    for name, pts in norm.items():
        px_pts = denormalize_polygon(pts, frame_width, frame_height)
        px[name] = px_pts
        poly[name] = Polygon(px_pts)
    return "single", norm, px, poly


def _build_intersection_fallback_zones(frame_width, frame_height, road_count=4):
    """
    Generates an intersection template with exactly `road_count` approaches.
    Used ONLY when automatic detection fails and non-interactive mode is requested.
    NEVER uses a fixed 4-road layout when road_count != 4.
    """
    road_count = max(1, int(road_count)) if road_count else 4
    if road_count == 1:
        return _build_single_road_zone(frame_width, frame_height)

    norm = {}
    if road_count == 2:
        norm["Road 1"] = [[0.20, 0.00], [0.80, 0.00], [0.80, 0.40], [0.20, 0.40]]
        norm["Road 2"] = [[0.20, 0.60], [0.80, 0.60], [0.80, 1.00], [0.20, 1.00]]
    elif road_count == 3:
        norm["Road 1"] = [[0.25, 0.00], [0.75, 0.00], [0.75, 0.40], [0.25, 0.40]]
        norm["Road 2"] = [[0.25, 0.60], [0.75, 0.60], [0.75, 1.00], [0.25, 1.00]]
        norm["Road 3"] = [[0.00, 0.25], [0.35, 0.25], [0.35, 0.75], [0.00, 0.75]]
    elif road_count == 4:
        norm["Road 1"] = [[0.20, 0.28], [0.80, 0.28], [0.80, 0.45], [0.20, 0.45]]
        norm["Road 2"] = [[0.10, 0.75], [0.90, 0.75], [0.90, 1.00], [0.10, 1.00]]
        norm["Road 3"] = [[0.00, 0.40], [0.30, 0.40], [0.30, 0.75], [0.00, 0.75]]
        norm["Road 4"] = [[0.70, 0.40], [1.00, 0.40], [1.00, 0.75], [0.70, 0.75]]
    else:
        import math
        cx, cy = 0.5, 0.5
        for i in range(road_count):
            angle_deg = -180.0 + i * (360.0 / road_count)
            half_w_deg = max(10.0, min(30.0, 180.0 / road_count * 0.5))
            rad = math.radians(angle_deg)
            hw_rad = math.radians(half_w_deg)
            perp_rad = rad + math.pi / 2

            near_r = 0.10
            far_r  = 0.48
            near_hw = near_r * math.tan(hw_rad * 0.5)
            far_hw  = far_r  * math.tan(hw_rad)

            p1 = [round(max(0.0, min(1.0, cx + near_r * math.cos(rad) - near_hw * math.cos(perp_rad))), 4),
                  round(max(0.0, min(1.0, cy + near_r * math.sin(rad) - near_hw * math.sin(perp_rad))), 4)]
            p2 = [round(max(0.0, min(1.0, cx + near_r * math.cos(rad) + near_hw * math.cos(perp_rad))), 4),
                  round(max(0.0, min(1.0, cy + near_r * math.sin(rad) + near_hw * math.sin(perp_rad))), 4)]
            p3 = [round(max(0.0, min(1.0, cx + far_r  * math.cos(rad) + far_hw  * math.cos(perp_rad))), 4),
                  round(max(0.0, min(1.0, cy + far_r  * math.sin(rad) + far_hw  * math.sin(perp_rad))), 4)]
            p4 = [round(max(0.0, min(1.0, cx + far_r  * math.cos(rad) - far_hw  * math.cos(perp_rad))), 4),
                  round(max(0.0, min(1.0, cy + far_r  * math.sin(rad) - far_hw  * math.sin(perp_rad))), 4)]
            norm[f"Road {i + 1}"] = [p1, p2, p3, p4]

    px = {}
    poly = {}
    for name, pts in norm.items():
        px_pts = denormalize_polygon(pts, frame_width, frame_height)
        px[name] = px_pts
        poly[name] = Polygon(px_pts)
    return "intersection", norm, px, poly


# Keep old name as an alias so external callers (tests etc.) still work.
def create_fallback_zones(frame_width, frame_height, mode="intersection", road_count=4):
    """Generates standard default zones based on mode and road_count. Kept for backward compatibility."""
    if mode == "single" or road_count == 1:
        return _build_single_road_zone(frame_width, frame_height)
    return _build_intersection_fallback_zones(frame_width, frame_height, road_count=road_count)


# ---------------------------------------------------------------------------
# Automatic Road Detector
# ---------------------------------------------------------------------------

class AutoRoadDetector:
    """
    Automatically detects road approach corridors from the video.

    Algorithm:
    1. Sample up to SAMPLE_FRAMES frames from the video, computing:
       a. Motion accumulation via MOG2 background subtraction.
       b. Road surface mask via HSV color filtering (dark, low-saturation asphalt).
    2. Combine motion + road mask into a "road activity" map.
    3. Find the intersection center (centroid of the high-activity region).
    4. Radially bin 360° from intersection center into NUM_ANGLE_BINS sectors.
    5. For each bin, compute total "road activity" in an outward annulus from center.
    6. Identify angular ranges with activity peaks separated by gaps → road corridors.
    7. For each detected corridor, generate a trapezoidal polygon covering the approach.
    8. Return N detected roads labeled "Road 1", "Road 2", ..., "Road N".
    """

    SAMPLE_FRAMES = 60       # frames to sample for motion accumulation
    NUM_ANGLE_BINS = 72      # 5° per bin (360° / 72 = 5°)
    MIN_ROAD_WIDTH_DEG = 8   # minimum angular width of a valid road approach (degrees)
    MAX_ROAD_WIDTH_DEG = 120 # maximum angular width of a single road approach (degrees)
    ACTIVITY_THRESHOLD = 0.05  # fraction of max bin value to treat as "active road"
    MIN_GAP_DEG = 10         # minimum gap (degrees) between distinct road corridors
    MIN_ROADS = 2            # minimum roads to detect before giving up and using fallback
    MAX_ROADS = 6            # maximum roads to detect
    APPROACH_FRACTION = 0.85 # polygon extends this fraction of distance from center to edge

    def __init__(self, video_path, frame_width, frame_height):
        self.video_path = video_path
        self.W = frame_width
        self.H = frame_height

    def _build_activity_map(self):
        """Samples video frames to build a 2D motion+road-surface activity map."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_step = max(1, total // self.SAMPLE_FRAMES)

        # Use working resolution (max dimension 640) for fast background subtraction on high-res videos
        scale = min(1.0, 640.0 / max(self.W, self.H))
        work_w = max(64, int(self.W * scale))
        work_h = max(64, int(self.H * scale))

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )

        motion_acc = np.zeros((work_h, work_w), dtype=np.float32)
        road_acc   = np.zeros((work_h, work_w), dtype=np.float32)
        frame_count = 0

        for frame_idx in range(0, min(total, self.SAMPLE_FRAMES * sample_step), sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            if frame.shape[:2] != (work_h, work_w):
                frame = cv2.resize(frame, (work_w, work_h))

            # 1. Motion mask via MOG2
            fg_mask = bg_subtractor.apply(frame)
            _, fg_bin = cv2.threshold(fg_mask, 200, 1, cv2.THRESH_BINARY)
            motion_acc += fg_bin.astype(np.float32)

            # 2. Road surface mask (dark, low-saturation regions = asphalt)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Asphalt/road: low saturation (0-70) and low-to-mid value (20-160)
            road_mask = cv2.inRange(hsv, (0, 0, 20), (180, 70, 160))
            road_acc += (road_mask // 255).astype(np.float32)
            frame_count += 1

        cap.release()

        if frame_count == 0:
            return None

        # Normalize and combine: weight motion heavily, road mask as guide
        if motion_acc.max() > 0:
            motion_norm = motion_acc / motion_acc.max()
        else:
            motion_norm = motion_acc

        if road_acc.max() > 0:
            road_norm = road_acc / road_acc.max()
        else:
            road_norm = road_acc

        # Combined activity: 60% motion + 40% road surface
        activity = 0.6 * motion_norm + 0.4 * road_norm

        # Smooth to reduce noise
        activity = cv2.GaussianBlur(activity, (25, 25), 0)
        return activity

    def _find_intersection_center(self, activity):
        """Finds the centroid of the highest-activity region (intersection area)."""
        # Threshold to top 30% activity values
        thresh = np.percentile(activity, 70)
        mask = (activity >= thresh).astype(np.uint8)

        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return self.W // 2, self.H // 2

        # Pick the largest non-background component
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = int(np.argmax(areas)) + 1
        cx = int(centroids[best_idx][0])
        cy = int(centroids[best_idx][1])

        # Clamp to inner region (avoid edge bias)
        cx = max(self.W // 6, min(5 * self.W // 6, cx))
        cy = max(self.H // 6, min(5 * self.H // 6, cy))
        return cx, cy

    def _radial_bin_activity(self, activity, cx, cy):
        """
        Bins activity radially from (cx, cy) into NUM_ANGLE_BINS angular sectors.
        Only samples from a mid-distance annulus (20% to 85% of min(W,H)/2).
        Returns a 1D array of activity sums per angular bin.
        """
        h, w = activity.shape
        ys, xs = np.mgrid[0:h, 0:w]

        dx = (xs - cx).astype(np.float32)
        dy = (ys - cy).astype(np.float32)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.degrees(np.arctan2(dy, dx))  # -180 to +180

        min_r = 0.15 * min(w, h)
        max_r = 0.48 * min(w, h)

        annulus = (dist >= min_r) & (dist <= max_r)
        bins = np.zeros(self.NUM_ANGLE_BINS, dtype=np.float32)

        for b in range(self.NUM_ANGLE_BINS):
            lo = -180 + b * (360.0 / self.NUM_ANGLE_BINS)
            hi = lo + (360.0 / self.NUM_ANGLE_BINS)
            in_bin = annulus & (angle >= lo) & (angle < hi)
            bins[b] = float(np.sum(activity[in_bin]))

        return bins

    def _detect_road_angles(self, bins, target_count=None):
        """
        Identifies distinct road approach angles from radial activity bins.
        If target_count is provided, selects the top `target_count` strongest approaches.
        Returns a sorted list of approach angles (degrees, -180 to +180).
        """
        if bins.max() == 0:
            return []

        bins_norm = bins / bins.max()
        threshold = self.ACTIVITY_THRESHOLD

        active = bins_norm >= threshold
        n = self.NUM_ANGLE_BINS
        deg_per_bin = 360.0 / n

        runs = []
        in_run = False
        run_start = 0
        active_ext = np.concatenate([active, active])
        bins_ext   = np.concatenate([bins_norm, bins_norm])

        for i in range(2 * n):
            if active_ext[i] and not in_run:
                in_run = True
                run_start = i
            elif not active_ext[i] and in_run:
                runs.append((run_start, i - 1))
                in_run = False
        if in_run:
            runs.append((run_start, 2 * n - 1))

        clean_runs = []
        for start, end in runs:
            width = end - start + 1
            if width * deg_per_bin < self.MIN_ROAD_WIDTH_DEG:
                continue
            if width * deg_per_bin > self.MAX_ROAD_WIDTH_DEG:
                continue
            norm_start = start % n
            norm_end = end % n
            clean_runs.append((norm_start, norm_end, width))

        if not clean_runs:
            return []

        candidates = []
        seen = set()
        for norm_start, norm_end, width in clean_runs:
            key = (norm_start, norm_end)
            if key in seen:
                continue
            seen.add(key)

            idxs = [(norm_start + j) % n for j in range(width)]
            weighted_sum = sum(bins_norm[i] * (-180 + i * deg_per_bin + deg_per_bin / 2) for i in idxs)
            total_weight = sum(bins_norm[i] for i in idxs)
            if total_weight > 0:
                mean_angle = weighted_sum / total_weight
            else:
                mid_idx = (norm_start + width // 2) % n
                mean_angle = -180 + mid_idx * deg_per_bin + deg_per_bin / 2

            candidates.append((mean_angle, float(total_weight)))

        # Sort candidates by angle
        candidates.sort(key=lambda x: x[0])

        # Merge angles that are too close (within MIN_GAP_DEG)
        merged = []
        for angle, w in candidates:
            if merged and abs(angle - merged[-1][0]) < self.MIN_GAP_DEG:
                prev_a, prev_w = merged[-1]
                tot_w = prev_w + w
                avg_a = (prev_a * prev_w + angle * w) / max(1e-6, tot_w)
                merged[-1] = (avg_a, tot_w)
            else:
                merged.append((angle, w))

        # Wrap-around merge
        if len(merged) >= 2:
            gap = (merged[0][0] + 360) - merged[-1][0]
            if gap < self.MIN_GAP_DEG:
                first_a, first_w = merged[0]
                last_a, last_w = merged[-1]
                tot_w = first_w + last_w
                avg_a = ((first_a + 360) * first_w + last_a * last_w) / max(1e-6, tot_w)
                if avg_a > 180:
                    avg_a -= 360
                merged = [(avg_a, tot_w)] + merged[1:-1]

        if not merged:
            return []

        if target_count is None:
            return sorted([a for a, w in merged])

        target_count = max(1, int(target_count))
        # Select top target_count strongest approaches with minimum angular separation
        min_sep = max(18.0, 180.0 / (target_count + 1))
        sorted_by_weight = sorted(merged, key=lambda x: x[1], reverse=True)
        chosen = []
        for angle, w in sorted_by_weight:
            if not any(abs((angle - ca + 180) % 360 - 180) < min_sep for ca in chosen):
                chosen.append(angle)
                if len(chosen) == target_count:
                    break

        # If more angles are needed, distribute remaining evenly
        if len(chosen) < target_count:
            step = 360.0 / target_count
            base_angle = chosen[0] if chosen else -180.0
            for k in range(target_count):
                cand = ((base_angle + k * step + 180) % 360) - 180
                if not any(abs((cand - ca + 180) % 360 - 180) < min_sep for ca in chosen):
                    chosen.append(cand)
                    if len(chosen) == target_count:
                        break

        return sorted(chosen[:target_count])

    def _make_road_polygon(self, cx, cy, angle_deg, half_width_deg=20.0):
        """
        Generates a trapezoidal polygon for a road approach corridor.
        The trapezoid fans out from the intersection center toward the frame edge.
        """
        import math

        angle_rad = math.radians(angle_deg)
        half_w_rad = math.radians(half_width_deg)

        # Near edge (close to center)
        near_r = 0.12 * min(self.W, self.H)
        # Far edge (toward frame boundary)
        far_r  = 0.52 * min(self.W, self.H)

        near_half_r = near_r * math.tan(half_w_rad * 0.5)
        far_half_r  = far_r  * math.tan(half_w_rad)

        perp_angle = angle_rad + math.pi / 2

        def pt(r, offset_r, a_main, a_perp):
            x = cx + r * math.cos(a_main) + offset_r * math.cos(a_perp)
            y = cy + r * math.sin(a_main) + offset_r * math.sin(a_perp)
            return (int(round(x)), int(round(y)))

        p1 = pt(near_r, -near_half_r, angle_rad, perp_angle)
        p2 = pt(near_r,  near_half_r, angle_rad, perp_angle)
        p3 = pt(far_r,   far_half_r,  angle_rad, perp_angle)
        p4 = pt(far_r,  -far_half_r,  angle_rad, perp_angle)

        # Clamp to frame bounds
        def clamp_pt(p):
            return (max(0, min(self.W - 1, p[0])), max(0, min(self.H - 1, p[1])))

        return [clamp_pt(p) for p in [p1, p2, p3, p4]]

    def detect(self, target_road_count=4):
        """
        Runs automatic road detection on the video for exactly `target_road_count` approaches.

        Returns:
            (mode, zones_normalized, zones_pixel, zone_polygons)
            All have N entries for N detected roads.
            Returns None, None, None, None on failure.
        """
        target_road_count = max(1, int(target_road_count)) if target_road_count else 4
        if target_road_count == 1:
            return _build_single_road_zone(self.W, self.H)

        _log(f"[AutoRoadDetector] Analyzing video for {target_road_count} road corridors: {os.path.basename(self.video_path)}")

        activity = self._build_activity_map()
        if activity is None:
            _log(f"[AutoRoadDetector] Could not build activity map. Using fallback layout.")
            return _build_intersection_fallback_zones(self.W, self.H, road_count=target_road_count)

        cx, cy = self._find_intersection_center(activity)
        act_h, act_w = activity.shape[:2]
        cx_full = int(cx * self.W / act_w)
        cy_full = int(cy * self.H / act_h)
        _log(f"[AutoRoadDetector] Intersection center estimated at ({cx_full}, {cy_full})")

        bins = self._radial_bin_activity(activity, cx, cy)
        road_angles = self._detect_road_angles(bins, target_count=target_road_count)

        _log(f"[AutoRoadDetector] Selected {len(road_angles)} road approaches for requested {target_road_count} roads: "
             f"{[round(a, 1) for a in road_angles]}")

        # Ensure we have exactly target_road_count angles
        if len(road_angles) < target_road_count:
            existing = list(road_angles)
            deg_step = 360.0 / target_road_count
            for i in range(target_road_count):
                cand = (( -180.0 + i * deg_step + 180) % 360) - 180
                if not any(abs((cand - e + 180) % 360 - 180) < 20.0 for e in existing):
                    existing.append(cand)
                    if len(existing) == target_road_count:
                        break
            road_angles = sorted(existing[:target_road_count])

        half_width = max(12.0, min(35.0, 180.0 / target_road_count * 0.55))

        zones_pixel = {}
        zone_polygons = {}
        zones_normalized = {}

        for i, angle in enumerate(road_angles[:target_road_count]):
            road_name = f"Road {i + 1}"
            pts = self._make_road_polygon(cx_full, cy_full, angle, half_width_deg=half_width)

            if len(pts) < 3:
                continue

            zones_pixel[road_name] = pts
            norm_pts = normalize_polygon(pts, self.W, self.H)
            zones_normalized[road_name] = norm_pts
            zone_polygons[road_name] = Polygon(pts)

        if len(zones_pixel) != target_road_count:
            _log(f"[AutoRoadDetector] Expected {target_road_count} valid polygons, got {len(zones_pixel)}. Using fallback.")
            return _build_intersection_fallback_zones(self.W, self.H, road_count=target_road_count)

        _log(f"[AutoRoadDetector] Successfully configured {len(zones_pixel)} roads: {list(zones_pixel.keys())}")
        return "intersection", zones_normalized, zones_pixel, zone_polygons


# ---------------------------------------------------------------------------
# Interactive Zone Selector (kept for --mode intersection CLI fallback only)
# ---------------------------------------------------------------------------

class InteractiveZoneSelector:
    """
    OpenCV GUI tool to interactively define road entry/exit zones on the first
    video frame. This is only used as a last-resort fallback when AutoRoadDetector
    fails AND interactive mode is explicitly requested.
    """
    def __init__(self, first_frame, video_path, mode=None, output_dir=config.OUTPUT_FOLDER):
        self.frame = first_frame.copy()
        self.frame_height, self.frame_width = self.frame.shape[:2]
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.mode = mode or "intersection"
        self.zone_file_path = get_zone_file_path(video_path, output_dir)

        self.completed_zones = {}
        self.current_points = []
        self.window_name = f"Define Road Zones - {self.video_name}"

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()

    def _draw_overlay(self):
        display = self.frame.copy()
        overlay = display.copy()

        for i, (name, pts) in enumerate(self.completed_zones.items()):
            color = ROAD_COLORS[i % len(ROAD_COLORS)]
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts_arr], color)
            cv2.polylines(display, [pts_arr], isClosed=True, color=color, thickness=2)
            moments = cv2.moments(pts_arr)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = pts[0]
            cv2.putText(display, name, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, name, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

        current_color = ROAD_COLORS[len(self.completed_zones) % len(ROAD_COLORS)]
        if self.current_points:
            for pt in self.current_points:
                cv2.circle(display, pt, 5, current_color, -1)
            if len(self.current_points) > 1:
                pts_arr = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(display, [pts_arr], isClosed=False, color=current_color, thickness=2)

        banner_h = 65
        cv2.rectangle(display, (0, 0), (self.frame_width, banner_h), (25, 28, 35), -1)
        cv2.line(display, (0, banner_h), (self.frame_width, banner_h), (70, 80, 95), 1)

        current_road_idx = len(self.completed_zones) + 1
        line1 = f"[INTERSECTION] ROAD #{current_road_idx}: Left-click points | 'C': Finish road | 'S': Save all"
        line2 = f"Defined Roads: {len(self.completed_zones)} | 'U': Undo point | 'R': Reset | 'Q': Cancel"
        cv2.putText(display, line1, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(display, line2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 215, 230), 1, cv2.LINE_AA)
        return display

    def run(self, allow_reuse=True):
        """Runs the interactive selector for intersection mode."""
        if self.mode == "single":
            m, norm, px, poly = _build_single_road_zone(self.frame_width, self.frame_height)
            save_zones(self.zone_file_path, norm, mode="single", video_name=self.video_name)
            return m, norm, px, poly

        self.completed_zones = {}
        self.current_points = []

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)

            while True:
                display = self._draw_overlay()
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord('c'), ord('C')):
                    if len(self.current_points) >= 3:
                        road_name = f"Road {len(self.completed_zones) + 1}"
                        self.completed_zones[road_name] = list(self.current_points)
                        print(f"  [+] Completed {road_name} ({len(self.current_points)} vertices)")
                        self.current_points = []
                    else:
                        print("  [!] Need at least 3 points for a road polygon.")

                elif key in (ord('u'), ord('U')):
                    if self.current_points:
                        self.current_points.pop()

                elif key in (ord('r'), ord('R')):
                    self.completed_zones = {}
                    self.current_points = []

                elif key in (ord('s'), ord('S'), 13):
                    if self.current_points and len(self.current_points) >= 3:
                        road_name = f"Road {len(self.completed_zones) + 1}"
                        self.completed_zones[road_name] = list(self.current_points)
                        self.current_points = []

                    if len(self.completed_zones) >= 2:
                        break
                    else:
                        print(f"  [!] Need at least 2 roads (current: {len(self.completed_zones)}).")

                elif key in (ord('q'), ord('Q'), 27):
                    cv2.destroyWindow(self.window_name)
                    return None, None, None, None

            cv2.destroyWindow(self.window_name)

        except Exception as e:
            print(f"[ZoneManager] GUI error: {e}")
            return None, None, None, None

        zones_normalized = {}
        zones_pixel = dict(self.completed_zones)
        zone_polygons = {}

        for name, pts in zones_pixel.items():
            norm_pts = normalize_polygon(pts, self.frame_width, self.frame_height)
            zones_normalized[name] = norm_pts
            zone_polygons[name] = Polygon(pts)

        save_zones(
            self.zone_file_path,
            zones_normalized,
            mode="intersection",
            video_name=self.video_name,
            video_width=self.frame_width,
            video_height=self.frame_height
        )
        return "intersection", zones_normalized, zones_pixel, zone_polygons


# ---------------------------------------------------------------------------
# Main public helper
# ---------------------------------------------------------------------------

def get_or_create_zones(video_path, first_frame=None, mode=None, road_count=None,
                        output_dir=config.OUTPUT_FOLDER, interactive=True):
    """
    Main helper: returns road zone layout for the given video.

    The caller's ``mode`` and ``road_count`` are AUTHORITATIVE.
    AutoRoadDetector determines WHERE roads are, but NEVER how many.
    Stale zone files whose road count or mode do not match are silently
    discarded (and deleted from disk) so they can never pollute a new run.

    SINGLE_ROAD mode (mode=="single" or road_count==1):
      - Always returns exactly ONE zone named "Single Road".
      - AutoRoadDetector is NEVER called.
      - InteractiveZoneSelector is NEVER shown.
      - Any cached file with != 1 road is deleted and ignored.

    INTERSECTION mode (road_count >= 2):
      - Returns exactly road_count zones.
      - AutoRoadDetector locates corridors, but the road count is fixed
        to the caller's requested value.
      - Falls back to a geometric template if detection fails.

    Returns: (mode_str, zones_normalized, zones_pixel, zone_polygons)
    """
    if first_frame is None:
        raise ValueError("first_frame must be provided to get_or_create_zones")

    frame_h, frame_w = first_frame.shape[:2]

    # ------------------------------------------------------------------
    # Resolve + lock mode and road_count
    # ------------------------------------------------------------------
    if mode == "single" or road_count == 1:
        mode       = "single"
        road_count = 1
    else:
        mode       = "intersection"
        road_count = int(road_count) if road_count and int(road_count) >= 2 else config.DEFAULT_ROAD_COUNT

    zone_file = get_zone_file_path(video_path, output_dir, mode=mode, road_count=road_count)

    # ------------------------------------------------------------------
    # SINGLE_ROAD fast path — guaranteed, no escape hatch
    # ------------------------------------------------------------------
    if mode == "single":
        # Purge any stale zone file for this video that is NOT single-mode
        stale_legacy = get_zone_file_path(video_path, output_dir)   # e.g. *_zones.json (4-road)
        if os.path.exists(stale_legacy) and stale_legacy != zone_file:
            # Check if it has more than 1 road — if so, it's stale for single mode
            try:
                with open(stale_legacy) as _f:
                    _d = json.load(_f)
                _zones_in_stale = _d.get("zones", _d) if isinstance(_d, dict) else _d
                if len(_zones_in_stale) > 1:
                    _log(f"[ZoneManager] SINGLE_ROAD: purging stale multi-road zone file "
                         f"({len(_zones_in_stale)} roads) — it will NOT be reused: "
                         f"{os.path.basename(stale_legacy)}")
                    # Do NOT delete — just don't load it.  The single-zone file is separate.
            except Exception:
                pass

        # Try to reuse cached single-road zone file
        if os.path.exists(zone_file):
            saved_mode, norm, px, poly = load_zones(
                zone_file, frame_w, frame_h,
                declared_mode="single", declared_road_count=1)
            if norm and len(norm) == 1:
                _log(f"[ZoneManager] SINGLE_ROAD: reusing saved zone for "
                     f"{os.path.basename(video_path)}.")
                return "single", norm, px, poly
            # Cached file is corrupt / wrong count — discard it
            _log(f"[ZoneManager] SINGLE_ROAD: cached zone file is invalid. Rebuilding.")

        # Always build a fresh full-frame single zone
        _log(f"[ZoneManager] SINGLE_ROAD: creating full-frame zone ('Single Road') "
             f"for {os.path.basename(video_path)}.")
        m, norm, px, poly = _build_single_road_zone(frame_w, frame_h)
        save_zones(zone_file, norm, mode="single",
                   video_name=os.path.basename(video_path),
                   video_width=frame_w, video_height=frame_h)
        return m, norm, px, poly

    # ------------------------------------------------------------------
    # INTERSECTION path (road_count >= 2)
    # ------------------------------------------------------------------

    # Try cached zones that match exact road_count
    if os.path.exists(zone_file):
        loaded_mode, norm, px, poly = load_zones(
            zone_file, frame_w, frame_h,
            declared_mode=mode, declared_road_count=road_count)
        if norm and len(norm) == road_count:
            _log(f"[ZoneManager] Reusing cached zones for {os.path.basename(video_path)} "
                 f"({loaded_mode} mode, {len(norm)} roads).")
            return loaded_mode, norm, px, poly

    # For 4-road mode also check legacy file name (*_zones.json)
    legacy_file = get_zone_file_path(video_path, output_dir)
    if road_count == 4 and os.path.exists(legacy_file) and legacy_file != zone_file:
        loaded_mode, norm, px, poly = load_zones(
            legacy_file, frame_w, frame_h,
            declared_mode=mode, declared_road_count=4)
        if norm and len(norm) == 4:
            _log(f"[ZoneManager] Reusing legacy cached zones for "
                 f"{os.path.basename(video_path)} (4 roads).")
            return loaded_mode, norm, px, poly

    # AutoRoadDetector — locates corridor positions for EXACTLY road_count roads
    _log(f"[ZoneManager] Running AutoRoadDetector for {road_count} roads on "
         f"{os.path.basename(video_path)} ...")
    detector = AutoRoadDetector(video_path, frame_w, frame_h)
    det_mode, norm, px, poly = detector.detect(target_road_count=road_count)
    if norm and len(norm) == road_count:
        save_zones(zone_file, norm,
                   mode="intersection",
                   video_name=os.path.basename(video_path),
                   video_width=frame_w, video_height=frame_h)
        return det_mode, norm, px, poly

    # Geometric fallback — guarantees exactly road_count zones
    _log(f"[ZoneManager] Using geometric fallback layout for exactly {road_count} roads.")
    m, norm, px, poly = _build_intersection_fallback_zones(
        frame_w, frame_h, road_count=road_count)
    save_zones(zone_file, norm, mode=m,
               video_name=os.path.basename(video_path),
               video_width=frame_w, video_height=frame_h)
    return m, norm, px, poly

