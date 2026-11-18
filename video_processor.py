"""
Video Processor pipeline coordinator module.
Executes frame-by-frame vehicle detection (YOLOv8), tracking (BoT-SORT),
counting, turning movement classification, Auto-ARIMA forecasting,
adaptive green signal timing, and video annotation output.
"""
import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd

import config
from detector import VehicleDetector
from tracker import VehicleTracker
from counter import VehicleCounter
from turning_detector import TurningDetector
from forecasting import TrafficForecaster
from signal_controller import AdaptiveSignalController
from traffic_state import TrafficState
import zone_manager

class VideoProcessor:
    def __init__(self, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD, output_dir=config.OUTPUT_FOLDER, mode=config.DEFAULT_MODE, road_count=config.DEFAULT_ROAD_COUNT):
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pipeline submodules
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker(self.detector)
        self.counter = VehicleCounter()
        self.turning_detector = TurningDetector()
        self.forecaster = TrafficForecaster(name="Overall")
        self.road_forecasters = {}
        self.signal_controller = AdaptiveSignalController()
        
        # Video capture & output paths
        self.cap = None
        self.writer = None
        self.video_path = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        
        # Dynamic road zones and mode
        self.mode = mode or "intersection"
        self.road_count = road_count or (1 if self.mode == "single" else config.DEFAULT_ROAD_COUNT)
        self.zones_normalized = {}
        self.zones_pixel = {}
        self.zone_polygons = {}
        
        self.output_video_path = None
        self.output_counts_csv = None
        self.output_forecast_csv = None
        self.output_summary_json = None
        
        # Latest stats snapshot for inspection
        self.latest_stats = {}
        
        # Class color palette for bounding boxes (BGR)
        self.class_colors = {
            0: (255, 180, 50),   # Car
            1: (0, 165, 255),    # Bus
            2: (0, 75, 255),     # Truck
            3: (255, 215, 0),    # Three Wheeler
            4: (0, 230, 115),    # Two Wheeler
            5: (220, 100, 220),  # LCV
            6: (240, 240, 50)    # Bicycle
        }

    def load_video(self, video_path, mode=None, interactive_zones=True):
        """Alias for open_video."""
        return self.open_video(video_path, mode=mode, interactive_zones=interactive_zones)

    # ------------------------------------------------------------------
    # MODE CONTRACT
    # ------------------------------------------------------------------
    # SINGLE_ROAD ("single"): ZoneManager is NOT used for zone drawing.
    #   A single full-frame zone is auto-created. No Road 1/2/3/4 logic.
    #   No turning movement calculation. One green time for the whole road.
    # INTERSECTION ("intersection"): ZoneManager interactive selector runs.
    #   User draws their own roads (3, 4, 5, 6+). Per-road signal timing.
    # ------------------------------------------------------------------

    def process_next_frame(self):
        """Alias for process_frame returning (ret, frame)."""
        ret, frame, _ = self.process_frame()
        return ret, frame

    def stop(self):
        """Alias for finalize."""
        return self.finalize()

    def get_stats(self):
        """Returns latest processing stats dictionary."""
        return dict(self.latest_stats)

    def open_video(self, video_path, mode=None, road_count=None, interactive_zones=True):
        """Validates video file, acquires road zones, initializes capture and output writer."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        allowed_exts = (".mp4", ".avi", ".mov", ".mkv")
        if not video_path.lower().endswith(allowed_exts):
            raise ValueError(f"Unsupported format. Allowed formats: {', '.join(allowed_exts)}")

        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()

        self.video_path = video_path
        video_stem = os.path.splitext(os.path.basename(video_path))[0]

        # ----------------------------------------------------------------
        # STEP 1: Resolve and LOCK mode + road_count from user's selection.
        # These values are authoritative.  Nothing downstream may override
        # them (not zone_manager, not AutoRoadDetector, not cached files).
        # ----------------------------------------------------------------
        if mode is not None:
            self.mode = mode
        if road_count is not None:
            self.road_count = road_count

        # Enforce single-road logic
        if self.mode == "single" or self.road_count == 1:
            self.mode       = "single"
            self.road_count = 1
        else:
            self.mode = "intersection"
            if not self.road_count or self.road_count < 2:
                self.road_count = config.DEFAULT_ROAD_COUNT

        # Freeze the user's authoritative values
        _auth_mode       = self.mode
        _auth_road_count = self.road_count

        # ----------------------------------------------------------------
        # STARTUP BANNER — printed only in debug mode
        # ----------------------------------------------------------------
        if getattr(config, "DEBUG", False):
            if _auth_mode == "single":
                print("\n" + "=" * 50)
                print("Mode: SINGLE ROAD")
                print("Road Count: 1")
                print("Multi-road processing: DISABLED")
                print("Turning detection: DISABLED")
                print("=" * 50 + "\n")
            else:
                print("\n" + "=" * 50)
                print("Mode: INTERSECTION")
                print(f"Road Count: {_auth_road_count}")
                print("Multi-road processing: ENABLED")
                print(f"Roads: {_auth_road_count}")
                print("=" * 50 + "\n")

        # Video-specific output file paths
        self.output_video_path  = os.path.join(self.output_dir, f"processed_{video_stem}.mp4")
        self.output_counts_csv  = os.path.join(self.output_dir, f"traffic_counts_{video_stem}.csv")
        self.output_forecast_csv = os.path.join(self.output_dir, f"forecast_{video_stem}.csv")
        self.output_summary_json = os.path.join(self.output_dir, f"summary_{video_stem}.json")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val           = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps          = fps_val if fps_val and fps_val > 0 else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        # Read first frame for zone initialisation
        ret, first_frame = self.cap.read()
        if not ret or first_frame is None:
            raise RuntimeError(f"Could not read first frame of video: {video_path}")

        # ----------------------------------------------------------------
        # STEP 2: Get road zones — zone_manager is given the authoritative
        # mode + road_count so it can never create extra roads.
        # ----------------------------------------------------------------
        m, norm, px, poly = zone_manager.get_or_create_zones(
            video_path=video_path,
            first_frame=first_frame,
            mode=_auth_mode,
            road_count=_auth_road_count,
            output_dir=self.output_dir,
            interactive=interactive_zones
        )
        if norm is None:
            raise RuntimeError("Zone selection was cancelled or no zones defined.")

        # ----------------------------------------------------------------
        # STEP 3: RE-ENFORCE authoritative mode + road_count.
        # Whatever zone_manager returned is used for zone geometry only.
        # Mode and road_count come exclusively from the user's selection.
        # ----------------------------------------------------------------
        self.mode              = _auth_mode          # user's selection wins
        self.zones_normalized  = norm or {}
        self.zones_pixel       = px   or {}
        self.zone_polygons     = poly or {}

        # For single-road mode, guarantee exactly 1 zone regardless of what
        # the zone file or detector returned.
        if _auth_mode == "single":
            self.road_count = 1
            # If zone_manager somehow returned multiple zones, truncate to 1
            if len(self.zones_pixel) != 1:
                if getattr(config, "DEBUG", False):
                    print("[VideoProcessor] WARNING: zone_manager returned "
                          f"{len(self.zones_pixel)} zones in SINGLE mode. "
                          "Overriding to single full-frame zone.")
                m2, norm2, px2, poly2 = zone_manager._build_single_road_zone(
                    self.frame_width, self.frame_height)
                self.zones_normalized = norm2
                self.zones_pixel      = px2
                self.zone_polygons    = poly2
        else:
            # Intersection: enforce the exact road count requested
            actual = len(self.zones_pixel)
            if actual != _auth_road_count:
                if getattr(config, "DEBUG", False):
                    print(f"[VideoProcessor] WARNING: zone_manager returned {actual} zones "
                          f"but road_count={_auth_road_count}. Using fallback layout.")
                m2, norm2, px2, poly2 = zone_manager._build_intersection_fallback_zones(
                    self.frame_width, self.frame_height, road_count=_auth_road_count)
                self.zones_normalized = norm2
                self.zones_pixel      = px2
                self.zone_polygons    = poly2
            self.road_count = _auth_road_count

        if getattr(config, "DEBUG", False):
            print(f"[VideoProcessor] Road processing enabled: {self.road_count}")

        # Rewind video to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ----------------------------------------------------------------
        # STEP 4: Initialise TrafficState with locked mode + road_count
        # ----------------------------------------------------------------
        self.traffic_state = TrafficState(
            mode=self.mode,
            zones_pixel=self.zones_pixel,
            zones_normalized=self.zones_normalized,
            zone_polygons=self.zone_polygons,
            road_count=self.road_count
        )

        # Re-initialise pipeline components for new session
        self.tracker         = VehicleTracker(self.detector)
        self.counter         = VehicleCounter()
        self.turning_detector = TurningDetector(self.zones_pixel, self.zone_polygons, mode=self.mode)
        self.forecaster      = TrafficForecaster(name="Overall")
        self.road_forecasters = {
            rname: rdata["forecaster"]
            for rname, rdata in self.traffic_state.roads.items()
        }
        self.signal_controller = AdaptiveSignalController()

        # Build Road ROI polygon
        self.roi_poly, self.roi_pts = self.detector.get_roi_polygon(
            self.frame_width, self.frame_height)

        # Initialise video output writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_video_path, fourcc, self.fps,
            (self.frame_width, self.frame_height)
        )


    def process_frame(self):
        """
        Executes pipeline on the next video frame:
        Detection -> BoT-SORT Tracking -> Central TrafficState update -> Visuals -> Side-by-side view.
        Returns (success, annotated_frame, stats_dict).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, None
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None
            
        self.current_frame_idx += 1
        frame_idx = self.current_frame_idx
        timestamp_sec = frame_idx / self.fps
        
        # 1. Detection + BoT-SORT Tracking + ROI Validation
        active_vehicles = self.tracker.update(
            frame,
            frame_idx,
            self.roi_poly,
            self.confidence_threshold
        )
        
        # 2. Update Central Traffic State (Single Source of Truth)
        self.traffic_state.update(
            active_vehicles,
            frame_idx,
            timestamp_sec,
            self.fps
        )
        
        current_vehicles = len(self.traffic_state.active_track_ids)
        total_counted = len(self.traffic_state.counted_track_ids)
        class_counts = dict(self.traffic_state.overall_class_counts)
        
        turning_counts = {
            "left": len(self.traffic_state.turning_counts["LEFT"]),
            "straight": len(self.traffic_state.turning_counts["STRAIGHT"]),
            "right": len(self.traffic_state.turning_counts["RIGHT"])
        }
        
        # Per-road signal breakdown.
        # Demand = cumulative vehicle_count per road + road-specific forecast (matches TrafficState formula).
        # TrafficState is the authoritative source of green_time; SC is used for phase cycling only.
        road_demands = {}
        for rname, rdata in self.traffic_state.roads.items():
            actual   = float(rdata["vehicle_count"])
            forecast = rdata["forecast"]   # road-specific ARIMA value
            if forecast is not None:
                road_demands[rname] = 0.7 * actual + 0.3 * float(forecast)
            else:
                road_demands[rname] = actual

        sc_results = self.signal_controller.calculate_roads(road_demands, fps=self.fps)

        road_signals = {}
        for rname, rdata in self.traffic_state.roads.items():
            sc = sc_results.get(rname, {})
            road_signals[rname] = {
                "vehicles":        rdata["vehicle_count"],
                # Use TrafficState's green_time (fixed formula) as authoritative
                "green_seconds":   rdata["green_time"],
                "forecast":        rdata["forecast"],
                "forecast_display": rdata["forecast_display"],
                "signal_state":    sc.get("signal_state", "RED"),
                "phase_remaining": sc.get("phase_remaining", 0)
            }

        # Overall green display — use the currently active road's green time
        active_road_name = list(road_demands.keys())[self.signal_controller.active_road_index % max(1, len(road_demands))] if road_demands else None
        first_rdata = list(self.traffic_state.roads.values())[0] if self.traffic_state.roads else {}
        active_rdata = self.traffic_state.roads.get(active_road_name, first_rdata) if active_road_name else first_rdata
        green_seconds = active_rdata.get("green_time", config.MIN_GREEN_LIGHT)
        sig_state = road_signals.get(active_road_name, {}).get("signal_state", "GREEN") if active_road_name else "GREEN"
        green_text = f"{sig_state}: {green_seconds} SECONDS"
        forecast_display = first_rdata.get("forecast_display", "Preparing forecast...")
        pred_next = first_rdata.get("forecast")
        
        # Mirror to counter and turning_detector for backward compatibility
        self.counter.counted_track_ids = self.traffic_state.counted_track_ids
        self.counter.history.append({
            "frame": frame_idx,
            "current_vehicles": current_vehicles,
            "total_counted": total_counted,
            "class_counts": class_counts
        })
        self.turning_detector.straight_count = turning_counts["straight"]
        self.turning_detector.left_count = turning_counts["left"]
        self.turning_detector.right_count = turning_counts["right"]
        self.turning_detector.active_road_counts = {
            rname: len(rdata["active_vehicle_ids"]) for rname, rdata in self.traffic_state.roads.items()
        }
        
        # Periodic cleanup of stale tracks
        if frame_idx % 60 == 0:
            self.tracker.prune_old_tracks(frame_idx)
            
        # 3. Render Visual Annotations on video frame (Section 22)
        # NOTE: video frame receives ONLY bounding boxes, labels, trajectories, turn tags.
        # No statistics, no forecast, no green time is overlaid on the video itself.
        annotated_frame = self._render_visuals(
            frame,
            active_vehicles
        )
        
        # Write frame to output video
        if self.writer is not None:
            self.writer.write(annotated_frame)
            
        stats = {
            "current_frame": frame_idx,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 1),
            "mode": self.mode,
            "road_count": self.road_count,
            "current_vehicles": current_vehicles,
            "total_counted": total_counted,
            "class_counts": class_counts,
            "turning_counts": turning_counts,
            "forecast_display": forecast_display,
            "forecast_text": forecast_display,
            "forecast_value": pred_next,
            "green_seconds": green_seconds,
            "green_text": green_text,
            "road_signals": road_signals
        }
        self.latest_stats = stats
        return True, annotated_frame, stats

    def _render_visuals(self, frame, active_vehicles):
        """
        Draws ONLY bounding boxes, tracking IDs, vehicle class names, small road zone
        outlines/labels, trajectory trails, and completed turn tags on the video frame.
        NO statistics, forecasts, green times, or signal states are drawn on the video.
        """
        annotated = frame.copy()

        # 1. Draw subtle road zone outlines and road name labels at centroids
        #    SINGLE ROAD: NO colored zones drawn (full-frame zone is invisible)
        #    INTERSECTION: draw each road's polygon with label
        if self.mode != "single" and self.traffic_state.roads:
            overlay = annotated.copy()
            for i, (name, rdata) in enumerate(self.traffic_state.roads.items()):
                pts = rdata.get("pixel_polygon")
                if not pts or len(pts) < 3:
                    continue
                color = zone_manager.ROAD_COLORS[i % len(zone_manager.ROAD_COLORS)]
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr], color)
                cv2.polylines(annotated, [pts_arr], isClosed=True, color=color, thickness=2)

                # Small road label at zone centroid
                cx, cy = rdata.get("centroid", (0, 0))
                if cx == 0 and cy == 0 and len(pts) > 0:
                    cx, cy = pts[0]
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.rectangle(annotated, (int(cx) - 4, int(cy) - th - 4),
                              (int(cx) + tw + 4, int(cy) + 4), (20, 20, 20), -1)
                cv2.putText(annotated, name, (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            # Blend zone fills subtly (15% opacity)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # 2. Draw tracked vehicles: bounding box, trajectory trail, class label, turn tag
        for vehicle in active_vehicles:
            coords = vehicle["bbox"]
            cls_id = vehicle["class_id"]
            track_id = vehicle["track_id"]
            class_name = vehicle["class_name"]
            color = self.class_colors.get(cls_id, (0, 255, 0))

            # Trajectory trail
            traj = vehicle.get("trajectory", [])
            if len(traj) >= 2:
                pts = np.array([[p["center_x"], p["center_y"]] for p in traj[-20:]], dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

            # Bounding box
            cv2.rectangle(annotated, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)

            # Turn tag appended ONLY after movement classification is complete
            movement = self.traffic_state.get_movement(track_id)
            turn_tag = f" [{movement}]" if movement else ""
            label = f"#{track_id} {class_name}{turn_tag}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            label_y = max(coords[1] - 4, th + 4)
            cv2.rectangle(annotated, (coords[0], label_y - th - 3),
                          (coords[0] + tw + 4, label_y + 3), color, -1)
            cv2.putText(annotated, label, (coords[0] + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    def render_side_by_side_view(self, annotated_frame, stats):
        """
        Builds the side-by-side display canvas:
          LEFT : processed video (bounding boxes, labels, trajectories, turn tags).
          RIGHT: plain traffic information panel.

        RIGHT panel layout (intersection mode):
          Vehicle Counts
          Turning Counts
          Auto-ARIMA Forecast   ← numerical values only; silent while warming up
          Road Signals & Control ← ALL detected roads; each shows only Vehicles + Green Time
        """
        h, w = annotated_frame.shape[:2]

        # Panel dimensions and font scales
        panel_w = 380 if w >= 1920 else 340
        font_scale_title = 0.50
        font_scale_text  = 0.40
        font_scale_small = 0.35

        roads    = self.traffic_state.roads
        num_roads = max(1, len(roads))

        # Canvas height: each road in the signals section needs name(16) + vehicles(14) + green(14) = 44px
        line_gap        = 17
        section_gap     = 16
        per_road_signal_h = 44   # name row + vehicles row + green time row

        # sections: header(24) + vehicle-counts(7*17+20=139) + sep(16)
        #           + turning(3*17+20=71 + sep 16) [INTERSECTION ONLY]
        #           + arima-title+roads(20 + num_roads*17) + sep(16)
        #           + signals-title+roads(20 + num_roads*44) + bottom(30)
        turning_block_h = (3 * line_gap + 20 + section_gap) if self.mode != "single" else 0
        arima_block_h   = 20 + num_roads * line_gap
        signals_block_h = 20 + (num_roads * per_road_signal_h if self.mode != "single" else line_gap)

        needed_h = (24 + 139 + section_gap
                    + turning_block_h
                    + arima_block_h   + section_gap
                    + signals_block_h + 30)

        canvas_h = max(h, needed_h)
        canvas = np.zeros((canvas_h, w + panel_w, 3), dtype=np.uint8)

        # LEFT: processed video
        canvas[:h, :w] = annotated_frame

        # RIGHT: dark panel
        canvas[:, w:] = (24, 28, 33)
        cv2.line(canvas, (w, 0), (w, canvas_h), (60, 70, 80), 1)

        tx = w + 16
        ty = 24

        # ── 1. Vehicle Counts ────────────────────────────────────────────
        cv2.putText(canvas, "Vehicle Counts", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 240, 255), 1, cv2.LINE_AA)
        ty += 20
        class_counts = stats.get("class_counts", {})
        for cname in ["Car", "Bus", "Truck", "Three Wheeler", "Two Wheeler", "LCV", "Bicycle"]:
            cv2.putText(canvas, f"{cname}: {class_counts.get(cname, 0)}",
                        (tx + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (210, 220, 225), 1, cv2.LINE_AA)
            ty += line_gap
        ty += 4
        cv2.line(canvas, (tx, ty), (tx + panel_w - 32, ty), (50, 55, 65), 1)
        ty += section_gap

        # ── 2. Turning Counts (Intersection only) ────────────────────────
        if self.mode != "single":
            cv2.putText(canvas, "Turning Counts", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 240, 255), 1, cv2.LINE_AA)
            ty += 20
            t_counts = stats.get("turning_counts", {})
            for t_label, t_key in [("Left", "left"), ("Straight", "straight"), ("Right", "right")]:
                cv2.putText(canvas, f"{t_label}: {t_counts.get(t_key, 0)}",
                            (tx + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (210, 220, 225), 1, cv2.LINE_AA)
                ty += line_gap
            ty += 4
            cv2.line(canvas, (tx, ty), (tx + panel_w - 32, ty), (50, 55, 65), 1)
            ty += section_gap

        # ── 3. Auto-ARIMA Forecast ────────────────────────────────────────
        # Show ONLY numerical forecast values. Display NOTHING while warming up.
        cv2.putText(canvas, "Auto-ARIMA Forecast", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 240, 255), 1, cv2.LINE_AA)
        ty += 20

        if self.mode == "single":
            # Single-road: one overall forecast value
            pred = stats.get("forecast_value")
            if pred is not None:
                cv2.putText(canvas, f"Next Traffic: {pred} vehicles",
                            (tx + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (80, 240, 180), 1, cv2.LINE_AA)
                ty += line_gap
            else:
                # Warming up — show nothing; reserve the line height so layout is stable
                ty += line_gap
        else:
            # Intersection: one line per road, numerical only
            any_forecast = False
            for rname, rdata in roads.items():
                pred = rdata.get("forecast")
                if pred is not None:
                    cv2.putText(canvas, f"{rname}: {pred} vehicles",
                                (tx + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (80, 240, 180), 1, cv2.LINE_AA)
                    any_forecast = True
                ty += line_gap
            # If no forecasts are ready yet we still consume the line heights above for a stable layout

        ty += 4
        cv2.line(canvas, (tx, ty), (tx + panel_w - 32, ty), (50, 55, 65), 1)
        ty += section_gap

        # ── 4. Road Signals & Control (Intersection) / Adaptive Green Signal (Single) ─────
        # Intersection: shows ALL detected roads. Each road: name, Vehicles, Green Time.
        # Single Road: shows ONE Adaptive Green Signal block.
        road_signals_in_stats = stats.get("road_signals", {})
        if self.mode == "single":
            cv2.putText(canvas, "Adaptive Green Signal", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 240, 255), 1, cv2.LINE_AA)
            ty += 20
            first_rdata = list(roads.values())[0] if roads else {}
            first_sig   = list(road_signals_in_stats.values())[0] if road_signals_in_stats else {}
            gt = first_sig.get("green_seconds", first_rdata.get("green_time", config.MIN_GREEN_LIGHT))
            cv2.putText(canvas, f"Green Time: {gt} sec",
                        (tx + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (80, 220, 140), 1, cv2.LINE_AA)
            ty += line_gap
        else:
            cv2.putText(canvas, "Road Signals & Control", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 240, 255), 1, cv2.LINE_AA)
            ty += 20
            for i, (rname, rdata) in enumerate(roads.items()):
                zone_color = zone_manager.ROAD_COLORS[i % len(zone_manager.ROAD_COLORS)]
                sig_info   = road_signals_in_stats.get(rname, {})
                gt         = sig_info.get("green_seconds", rdata.get("green_time", config.MIN_GREEN_LIGHT))
                veh_cnt    = rdata.get("vehicle_count", 0)

                # Road name with colored zone dot
                cv2.circle(canvas, (tx + 8, ty - 4), 5, zone_color, -1)
                cv2.putText(canvas, rname, (tx + 18, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_text + 0.02, (255, 255, 255), 1, cv2.LINE_AA)
                ty += 16

                # Vehicles
                cv2.putText(canvas, f"Vehicles: {veh_cnt}",
                            (tx + 18, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (180, 220, 240), 1, cv2.LINE_AA)
                ty += 14

                # Green Time
                cv2.putText(canvas, f"Green Time: {gt} sec",
                            (tx + 18, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (80, 220, 140), 1, cv2.LINE_AA)
                ty += 14

        return canvas

    def finalize(self):
        """Releases video resources and exports CSV and JSON output files."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
        # 1. Save traffic counts CSV
        if self.counter.history:
            flat_rows = []
            for item in self.counter.history:
                row = {
                    "frame": item["frame"],
                    "current_vehicles": item["current_vehicles"],
                    "total_counted": item["total_counted"]
                }
                for cname, cval in item["class_counts"].items():
                    row[f"cum_{cname.lower().replace(' ', '_')}"] = cval
                flat_rows.append(row)
            pd.DataFrame(flat_rows).to_csv(self.output_counts_csv, index=False)
            
        # 2. Save Auto-ARIMA forecast CSV (with road column for all roads)
        combined_rows = []
        for rname, rdata in self.traffic_state.roads.items():
            rf = rdata["forecaster"]
            for obs in rf.observations:
                combined_rows.append({
                    "road": rname,
                    "interval_id": obs["interval_id"],
                    "timestamp_sec": obs["timestamp_sec"],
                    "vehicle_count": obs["vehicle_count"],
                    "forecast_next": obs.get("forecast_next") if obs.get("forecast_next") is not None else "N/A",
                    "arima_order": obs.get("arima_order") if obs.get("arima_order") is not None else "N/A"
                })
        if combined_rows:
            pd.DataFrame(combined_rows).to_csv(self.output_forecast_csv, index=False)
        else:
            pd.DataFrame(columns=["road", "interval_id", "timestamp_sec", "vehicle_count", "forecast_next", "arima_order"]).to_csv(self.output_forecast_csv, index=False)
            
        # 3. Save Summary JSON
        # Use the most-recent road_signals from latest_stats for accurate signal states
        last_road_signals = self.latest_stats.get("road_signals", {})
        roads_summary = {}
        for rname, rdata in self.traffic_state.roads.items():
            last_sig = last_road_signals.get(rname, {})
            roads_summary[rname] = {
                "vehicles": rdata["vehicle_count"],
                "green_seconds": rdata["green_time"],
                "forecast": rdata["forecast"],
                "forecast_display": rdata["forecast_display"],
                "signal_state": last_sig.get("signal_state", "RED"),
                "phase_remaining": last_sig.get("phase_remaining", 0)
            }
            
        road_forecasts = {}
        for rname, rdata in self.traffic_state.roads.items():
            rf = rdata["forecaster"]
            road_forecasts[rname] = {
                "final_forecast": rdata["forecast"],
                "status": rf.get_final_status(),
                "arima_order": str(rf.last_model_order) if rf.last_model_order else "N/A",
                "total_intervals_collected": len(rf.observations),
                "min_intervals_required": rf.min_observations
            }
            
        first_rdata = list(self.traffic_state.roads.values())[0] if self.traffic_state.roads else {}
        summary = {
            "video_metadata": {
                "source_path": self.video_path,
                "mode": self.mode,
                "road_count": self.road_count,
                "resolution": f"{self.frame_width}x{self.frame_height}",
                "fps": round(self.fps, 2),
                "total_frames_processed": self.current_frame_idx
            },
            "cumulative_vehicle_counts": dict(self.traffic_state.overall_class_counts),
            "total_unique_vehicles": len(self.traffic_state.counted_track_ids),
            "turning_patterns": {
                "straight": len(self.traffic_state.turning_counts["STRAIGHT"]),
                "left_turn": len(self.traffic_state.turning_counts["LEFT"]),
                "right_turn": len(self.traffic_state.turning_counts["RIGHT"]),
                "left": len(self.traffic_state.turning_counts["LEFT"]),
                "right": len(self.traffic_state.turning_counts["RIGHT"])
            } if self.mode == "intersection" else {
                "turning": "Not applicable"
            },
            "forecasting": {
                "final_forecast": first_rdata.get("forecast"),
                "status": first_rdata.get("forecast_display", "Preparing forecast..."),
                "arima_order": str(first_rdata["forecaster"].last_model_order) if "forecaster" in first_rdata and first_rdata["forecaster"].last_model_order else "N/A",
                "total_intervals_collected": len(first_rdata["forecaster"].observations) if "forecaster" in first_rdata else 0,
                "min_intervals_required": first_rdata["forecaster"].min_observations if "forecaster" in first_rdata else config.MIN_FORECAST_OBSERVATIONS,
                "road_forecasts": road_forecasts
            },
            "signal_timing": {
                "mode": self.mode,
                "recommended_green_seconds": first_rdata.get("green_time", config.MIN_GREEN_LIGHT) if self.mode == "single" else None,
                "road_signals": roads_summary
            },
            "output_files": {
                "processed_video": self.output_video_path,
                "traffic_counts_csv": self.output_counts_csv,
                "forecast_csv": self.output_forecast_csv,
                "summary_json": self.output_summary_json
            }
        }
        
        with open(self.output_summary_json, "w") as f:
            json.dump(summary, f, indent=4)
            
        return summary

    def process_video(self, video_path, max_frames=None, show=False, mode=None, road_count=None, interactive_zones=True):
        """
        Processes entire video file from start to finish.
        Displays terminal progress, optional side-by-side live window, and returns final summary.
        """
        self.open_video(video_path, mode=mode, road_count=road_count, interactive_zones=interactive_zones)
        
        limit_frames = max_frames if max_frames else self.total_frames
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Frames to process: {limit_frames}")
        print("-" * 45)
        
        start_time = time.time()
        
        while self.cap.isOpened():
            if max_frames and self.current_frame_idx >= max_frames:
                break
                
            ret, frame, stats = self.process_frame()
            if not ret:
                break
                
            frame_num = stats["current_frame"]
            pct = int((frame_num / max(1, limit_frames)) * 100)
            
            # Print live console progress line
            if self.mode == "intersection":
                t_str = f"S:{stats['turning_counts']['straight']} L:{stats['turning_counts']['left']} R:{stats['turning_counts']['right']} | "
                r_greens = " ".join(f"{r}:{info.get('vehicles', 0)}v/{info.get('green_seconds', config.MIN_GREEN_LIGHT)}s" for r, info in stats.get('road_signals', {}).items())
                status_signal = f"Roads: [{r_greens}]"
            else:
                t_str = ""
                status_signal = f"Green: {stats['green_seconds']:2d}s"
                
            f_val = stats.get("forecast_value")
            if f_val is not None:
                f_str = f"Next: {f_val}"
            else:
                f_str = f"Warmup ({len(self.forecaster.observations)}/{self.forecaster.min_observations})"
                
            sys.stdout.write(
                f"\r[Processing] Frame {frame_num:04d}/{limit_frames} ({pct:3d}%) | "
                f"Active: {stats['current_vehicles']:2d} | "
                f"Total: {stats['total_counted']:2d} | "
                f"{t_str}"
                f"{status_signal} | "
                f"Forecast: {f_str}"
            )
            sys.stdout.flush()
            
            # Side-by-side OpenCV window: LEFT = processed video, RIGHT = plain traffic info
            if show:
                display_frame = self.render_side_by_side_view(frame, stats)
                cv2.imshow("Traffic Video Analyzer", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user.")
                    break
                    
        sys.stdout.write("\n")
        if show:
            cv2.destroyAllWindows()
            
        elapsed = time.time() - start_time
        summary = self.finalize()
        summary["execution_time_seconds"] = round(elapsed, 2)
        return summary


