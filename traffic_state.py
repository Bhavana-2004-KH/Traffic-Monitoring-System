"""
Central Traffic State module for Standalone Traffic Video Processing Application.
Provides a unified, single source of truth for:
- Roads geometry, counting, forecasting, and signal green time
- Vehicle tracking, unique counting, and road assignments
- Trajectory analysis, completed turning classification, and traceability
"""
import math
from collections import defaultdict
import numpy as np
from shapely.geometry import Point, Polygon
import config
from forecasting import TrafficForecaster

class TrafficState:
    """
    Central traffic monitoring state.
    Shared across vehicle counting, road assignment, turning detection,
    Auto-ARIMA forecasting, green-time calculation, and UI rendering.
    """

    def __init__(self, mode="intersection", zones_pixel=None, zones_normalized=None, zone_polygons=None, road_count=None):
        self.mode = mode.lower() if mode else "intersection"
        self.zones_pixel = zones_pixel or {}
        self.zones_normalized = zones_normalized or {}
        self.zone_polygons = zone_polygons or {}

        # Road Count configuration
        if self.mode == "single" or road_count == 1:
            self.mode = "single"
            self.road_count = 1
        else:
            self.road_count = road_count or len(self.zones_pixel) or config.DEFAULT_ROAD_COUNT

        # -------------------------------------------------------------
        # Single Source of Truth: ROADS
        # -------------------------------------------------------------
        self.roads = {}
        self.intersection_center = None
        self._init_roads()

        # -------------------------------------------------------------
        # Single Source of Truth: VEHICLES
        # -------------------------------------------------------------
        self.vehicles = {}

        # -------------------------------------------------------------
        # Overall Unique Vehicle Counting
        # -------------------------------------------------------------
        self.counted_track_ids = set()
        self.overall_class_counts = {cname: 0 for cname in config.DISPLAY_NAMES.values()}

        # -------------------------------------------------------------
        # Turning Counts (Track IDs sets for traceability)
        # -------------------------------------------------------------
        self.turning_counts = {
            "LEFT": set(),
            "STRAIGHT": set(),
            "RIGHT": set()
        }

        # -------------------------------------------------------------
        # Active Sets for Status Tracking
        # -------------------------------------------------------------
        self.active_track_ids = set()
        self.completed_turn_ids = set()
        self.unclassified_active_ids = set()

        # Cycle tracking for green time updates
        self.cycle_frame_counter = 0

    def _init_roads(self):
        """Initializes the central roads data structure from configured zones."""
        self.roads.clear()

        if self.mode == "single" or self.road_count == 1 or not self.zones_pixel:
            # Single Road Mode: creates exactly one road named "Single Road"
            fc = TrafficForecaster(
                interval_seconds=config.FORECAST_INTERVAL_SECONDS,
                min_observations=config.MIN_FORECAST_OBSERVATIONS,
                name="Single Road"
            )
            # Support both "Single Road" and legacy "Road 1" zone keys
            pts      = (self.zones_pixel.get("Single Road")
                        or self.zones_pixel.get("Road 1")
                        or [])
            norm_pts = (self.zones_normalized.get("Single Road")
                        or self.zones_normalized.get("Road 1")
                        or [])
            poly     = (self.zone_polygons.get("Single Road")
                        or self.zone_polygons.get("Road 1")
                        or (Polygon(pts) if len(pts) >= 3 else None))

            self.roads["Single Road"] = {
                "id":               "Single Road",
                "name":             "Single Road",
                "pixel_polygon":    pts,
                "norm_polygon":     norm_pts,
                "polygon":          poly,
                "centroid":         (0, 0),
                "vehicle_ids":      set(),
                "active_vehicle_ids": set(),
                "vehicle_count":    0,
                "class_counts":     {cname: 0 for cname in config.DISPLAY_NAMES.values()},
                "forecast":         None,
                "forecast_display": "Preparing forecast...",
                "green_time":       config.MIN_GREEN_LIGHT,
                "forecaster":       fc,
                "prev_demand":      0.0,
                "prev_green_time":  config.MIN_GREEN_LIGHT
            }
            self.intersection_center = None
            return


        # Intersection Mode: Exactly all detected roads
        cx_list, cy_list = [], []
        for rname, pts in self.zones_pixel.items():
            pts_arr = np.array(pts, dtype=np.float32)
            c_x = float(np.mean(pts_arr[:, 0]))
            c_y = float(np.mean(pts_arr[:, 1]))
            cx_list.append(c_x)
            cy_list.append(c_y)

            poly = self.zone_polygons.get(rname)
            if poly is None and len(pts) >= 3:
                poly = Polygon(pts)

            fc = TrafficForecaster(
                interval_seconds=config.FORECAST_INTERVAL_SECONDS,
                min_observations=config.MIN_FORECAST_OBSERVATIONS,
                name=rname
            )

            self.roads[rname] = {
                "id": rname,
                "name": rname,
                "pixel_polygon": pts,
                "norm_polygon": self.zones_normalized.get(rname, []),
                "polygon": poly,
                "centroid": (c_x, c_y),
                "vehicle_ids": set(),
                "active_vehicle_ids": set(),
                "vehicle_count": 0,
                "class_counts": {cname: 0 for cname in config.DISPLAY_NAMES.values()},
                "forecast": None,
                "forecast_display": "Preparing forecast...",
                "green_time": config.MIN_GREEN_LIGHT,
                "forecaster": fc,
                "prev_demand": 0.0,
                "prev_green_time": config.MIN_GREEN_LIGHT
            }

        if cx_list and cy_list:
            self.intersection_center = (float(np.mean(cx_list)), float(np.mean(cy_list)))
        else:
            self.intersection_center = None

    def identify_road(self, point):
        """Returns the road name containing the point, or None."""
        if not self.roads:
            return None
        p = Point(point[0], point[1])
        for rname, rdata in self.roads.items():
            poly = rdata["polygon"]
            if poly and poly.contains(p):
                return rname
        return None

    def update(self, active_detections, frame_idx, timestamp_sec, fps):
        """
        Main per-frame state update.
        Synchronizes vehicle tracking, unique counting, road assignment,
        turn completion, time-series forecasting, and green time.
        """
        current_frame_tids = set()
        
        # 1. Process active confirmed detections
        for det in active_detections:
            if not det.get("confirmed", False):
                continue

            track_id = det["track_id"]
            if track_id is None:
                continue

            current_frame_tids.add(track_id)
            cls_id = det["class_id"]
            class_name = det.get("class_name") or config.DISPLAY_NAMES.get(cls_id, f"Class_{cls_id}")
            cx, cy = det["center"]

            # Initialize vehicle record if new
            if track_id not in self.vehicles:
                self.vehicles[track_id] = {
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "trajectory": [(cx, cy, frame_idx)],
                    "current_road": None,
                    "previous_road": None,
                    "entry_road": None,
                    "exit_road": None,
                    "turn": None,
                    "counted": False,
                    "movement_completed": False
                }
            else:
                self.vehicles[track_id]["last_seen"] = frame_idx
                self.vehicles[track_id]["trajectory"].append((cx, cy, frame_idx))

            v_record = self.vehicles[track_id]

            # Overall Unique Vehicle Count (counted once)
            if track_id not in self.counted_track_ids:
                self.counted_track_ids.add(track_id)
                self.overall_class_counts[class_name] = self.overall_class_counts.get(class_name, 0) + 1
                v_record["counted"] = True

            # Road Assignment based on polygon intersection
            if self.mode == "single":
                v_record["current_road"] = "Single Road"
                v_record["entry_road"]   = "Single Road"
                if track_id not in self.roads["Single Road"]["vehicle_ids"]:
                    self.roads["Single Road"]["vehicle_ids"].add(track_id)
                    self.roads["Single Road"]["vehicle_count"] = len(self.roads["Single Road"]["vehicle_ids"])
                    self.roads["Single Road"]["class_counts"][class_name] = self.roads["Single Road"]["class_counts"].get(class_name, 0) + 1

            else:
                road_hit = self.identify_road((cx, cy))
                v_record["previous_road"] = v_record["current_road"]
                v_record["current_road"] = road_hit

                if road_hit is not None:
                    # Vehicle is inside road_hit polygon
                    if track_id not in self.roads[road_hit]["vehicle_ids"]:
                        self.roads[road_hit]["vehicle_ids"].add(track_id)
                        self.roads[road_hit]["vehicle_count"] = len(self.roads[road_hit]["vehicle_ids"])
                        self.roads[road_hit]["class_counts"][class_name] = self.roads[road_hit]["class_counts"].get(class_name, 0) + 1
                        if getattr(config, "DEBUG", False):
                            print(f"{road_hit}: vehicle {track_id} added")

                    if v_record["entry_road"] is None:
                        v_record["entry_road"] = road_hit
                        if getattr(config, "DEBUG", False):
                            print(f"Track {track_id} | {class_name} | Entered: {road_hit}")
                    elif v_record["entry_road"] != road_hit and v_record["exit_road"] is None:
                        # Vehicle moved from entry road to a different road (exit road)
                        v_record["exit_road"] = road_hit

                # Check for turning classification (Section 7, 8, 11)
                # DO NOT classify until movement completes
                if not v_record["movement_completed"]:
                    traj = v_record["trajectory"]
                    entry = v_record["entry_road"]
                    exit_r = v_record["exit_road"]

                    # Path A: Vehicle entered an exit road different from entry road
                    if entry and exit_r and entry != exit_r and len(traj) >= 6:
                        turn_result = self.classify_turn_geometry(entry, exit_r, traj)
                        if turn_result:
                            self._register_completed_turn(v_record, turn_result, entry, exit_r)

                    # Path B: Kinematic trajectory analysis if vehicle traveled through intersection
                    elif entry and len(traj) >= 20 and not v_record["movement_completed"]:
                        turn_result, inferred_exit = self.classify_turn_trajectory(entry, traj)
                        if turn_result:
                            if inferred_exit and inferred_exit in self.roads:
                                if track_id not in self.roads[inferred_exit]["vehicle_ids"]:
                                    self.roads[inferred_exit]["vehicle_ids"].add(track_id)
                                    self.roads[inferred_exit]["vehicle_count"] = len(self.roads[inferred_exit]["vehicle_ids"])
                                    if getattr(config, "DEBUG", False):
                                        print(f"{inferred_exit}: vehicle {track_id} added")
                                v_record["exit_road"] = inferred_exit
                            final_exit = v_record["exit_road"] or "Exit"
                            self._register_completed_turn(v_record, turn_result, entry, final_exit)

        # 2. Update active sets
        self.active_track_ids = current_frame_tids
        self.completed_turn_ids = (
            self.turning_counts["LEFT"]
            | self.turning_counts["STRAIGHT"]
            | self.turning_counts["RIGHT"]
        )
        self.unclassified_active_ids = self.active_track_ids - self.completed_turn_ids

        # 3. Update per-road active vehicles
        for rname, rdata in self.roads.items():
            rdata["active_vehicle_ids"] = {
                tid for tid in self.active_track_ids
                if self.vehicles.get(tid, {}).get("current_road") == rname
            }

        # 4. Update Time-Series Auto-ARIMA Forecasting (Section 15, 16)
        # Exactly 5.0s intervals; runs after 2 intervals (10s)
        for rname, rdata in self.roads.items():
            fc = rdata["forecaster"]
            active_cnt = len(rdata["active_vehicle_ids"])
            fc.add_frame_sample(timestamp_sec, active_cnt, active_track_ids=rdata["active_vehicle_ids"])

            prev_pred = rdata["forecast"]
            new_pred = fc.predicted_next
            rdata["forecast"] = new_pred
            rdata["forecast_display"] = fc.get_forecast_display()

            if new_pred is not None and new_pred != prev_pred:
                hist = [obs["vehicle_count"] for obs in fc.observations[-5:]]
                if getattr(config, "DEBUG", False):
                    print(f"{rname}: history = {hist} -> forecast = {new_pred}")

        # 5. Update Adaptive Green Time (per road, independently)
        # Recalculates once per SIGNAL_CYCLE_FRAMES boundary to prevent flickering.
        self.cycle_frame_counter += 1
        if self.cycle_frame_counter >= config.SIGNAL_CYCLE_FRAMES:
            self.cycle_frame_counter = 0
            max_ch = getattr(config, "MAX_GREEN_CHANGE_PER_CYCLE", 5)

            if getattr(config, "DEBUG", False):
                print("\nGreen-time update:\n")
            for rname, rdata in self.roads.items():
                # --- Road-specific demand ---
                # Use cumulative vehicle_count (stable, per-road) as the actual demand.
                # If the road's own ARIMA forecast is available, blend it in.
                actual_count   = float(rdata["vehicle_count"])
                road_forecast  = rdata["forecast"]   # per-road Auto-ARIMA value (or None)

                if road_forecast is not None:
                    # 70 % actual traffic + 30 % road-specific forecast
                    demand = 0.7 * actual_count + 0.3 * float(road_forecast)
                else:
                    demand = actual_count

                # EMA smoothing: 0.7 * previous + 0.3 * current demand
                smoothed = 0.7 * rdata["prev_demand"] + 0.3 * demand
                rdata["prev_demand"] = smoothed

                # Formula: MIN_GREEN + demand * scale_factor
                #   0 vehicles  -> MIN_GREEN (20 s)
                #   10 vehicles -> 20 + 10*2.5 = 45 s
                #   26 vehicles -> 20 + 26*2.5 = 85 s  (near MAX)
                raw    = config.MIN_GREEN_LIGHT + (smoothed * config.SECONDS_PER_VEHICLE)
                target = int(round(max(config.MIN_GREEN_LIGHT,
                                       min(config.MAX_GREEN_LIGHT, raw))))

                # Rate-limit: never jump more than MAX_GREEN_CHANGE_PER_CYCLE per update
                prev_gt = rdata["green_time"]
                delta   = target - prev_gt
                if delta > max_ch:
                    new_gt = prev_gt + max_ch
                elif delta < -max_ch:
                    new_gt = prev_gt - max_ch
                else:
                    new_gt = target

                rdata["green_time"] = new_gt

                # Debug logging (requirement 18)
                if getattr(config, "DEBUG", False):
                    fc_val = int(road_forecast) if road_forecast is not None else "N/A"
                    print(f"{rname}:")
                    print(f"actual = {int(actual_count)}")
                    print(f"forecast = {fc_val}")
                    print(f"demand = {int(round(demand)) if abs(demand - round(demand)) < 1e-4 else round(demand, 1)}")
                    print(f"green = {new_gt} sec\n")

    def classify_turn_geometry(self, entry_road, exit_road, traj):
        """Classifies turn based on entry road and exit road centroids."""
        if not self.intersection_center or entry_road not in self.roads or exit_road not in self.roads:
            return "STRAIGHT"

        c_entry = self.roads[entry_road]["centroid"]
        c_exit = self.roads[exit_road]["centroid"]
        c_center = self.intersection_center

        # Vector entering from entry road to center
        v_in = np.array([c_center[0] - c_entry[0], c_center[1] - c_entry[1]], dtype=np.float64)
        # Vector exiting from center to exit road
        v_out = np.array([c_exit[0] - c_center[0], c_exit[1] - c_center[1]], dtype=np.float64)

        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        if norm_in < 1e-3 or norm_out < 1e-3:
            return "STRAIGHT"

        theta_in = math.atan2(v_in[1], v_in[0])
        theta_out = math.atan2(v_out[1], v_out[0])

        diff_deg = math.degrees(theta_out - theta_in)
        while diff_deg > 180:
            diff_deg -= 360
        while diff_deg < -180:
            diff_deg += 360

        # Screen coordinates: clockwise = positive = RIGHT, counter-clockwise = negative = LEFT
        if abs(diff_deg) <= 40:
            return "STRAIGHT"
        elif diff_deg > 40:
            return "RIGHT"
        else:
            return "LEFT"

    def classify_turn_trajectory(self, entry_road, traj):
        """
        Classifies turn using multi-point trajectory displacement deflection.
        Used when a vehicle traverses the intersection without explicitly entering a second road polygon.
        """
        start_pt = np.array([traj[0][0], traj[0][1]], dtype=np.float64)
        mid_pt = np.array([traj[len(traj) // 2][0], traj[len(traj) // 2][1]], dtype=np.float64)
        end_pt = np.array([traj[-1][0], traj[-1][1]], dtype=np.float64)

        total_disp = np.linalg.norm(end_pt - start_pt)
        if total_disp < 35:
            return None, None

        v1 = mid_pt - start_pt
        v2 = end_pt - mid_pt
        if np.linalg.norm(v1) < 15 or np.linalg.norm(v2) < 15:
            return None, None

        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        diff_deg = math.degrees(angle2 - angle1)

        while diff_deg > 180:
            diff_deg -= 360
        while diff_deg < -180:
            diff_deg += 360

        # Identify nearest road to end_pt as inferred exit
        best_road = None
        min_dist = float("inf")
        for rname, rdata in self.roads.items():
            if rname == entry_road:
                continue
            rc = rdata["centroid"]
            dist = math.hypot(end_pt[0] - rc[0], end_pt[1] - rc[1])
            if dist < min_dist:
                min_dist = dist
                best_road = rname

        if abs(diff_deg) <= 30:
            return "STRAIGHT", best_road
        elif diff_deg > 30:
            return "RIGHT", best_road
        else:
            return "LEFT", best_road

    def _register_completed_turn(self, v_record, turn_result, entry_road, exit_road):
        """Registers a completed turning event with full traceability to track ID."""
        track_id = v_record["track_id"]
        v_record["turn"] = turn_result
        v_record["movement_completed"] = True
        self.turning_counts[turn_result].add(track_id)
        if getattr(config, "DEBUG", False):
            print(f"Track {track_id} | {v_record['class_name']} | {entry_road} -> {exit_road} | {turn_result}")

    def get_movement(self, track_id):
        """Returns completed turn label ('LEFT', 'RIGHT', 'STRAIGHT') for a vehicle, or None."""
        if self.mode == "single":
            return None
        v = self.vehicles.get(track_id)
        if v and v.get("movement_completed"):
            return v.get("turn")
        return None

    def get_summary(self):
        """Returns comprehensive summary dictionary of the entire traffic state."""
        return {
            "mode": self.mode,
            "total_unique_vehicles": len(self.counted_track_ids),
            "class_counts": dict(self.overall_class_counts),
            "turning_counts": {
                "left": len(self.turning_counts["LEFT"]),
                "straight": len(self.turning_counts["STRAIGHT"]),
                "right": len(self.turning_counts["RIGHT"])
            },
            "completed_turn_ids": list(self.completed_turn_ids),
            "roads": {
                rname: {
                    "name": rname,
                    "vehicles": rdata["vehicle_count"],
                    "green_seconds": rdata["green_time"],
                    "forecast": rdata["forecast"],
                    "forecast_display": rdata["forecast_display"]
                }
                for rname, rdata in self.roads.items()
            }
        }
