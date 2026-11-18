"""
Tracker module using BoT-SORT for robust multi-vehicle tracking.
Maintains vehicle trajectories, tracking IDs, and temporal consistency.
"""
from collections import defaultdict, deque
import cv2
import numpy as np
import config

class VehicleTracker:
    def __init__(self, detector, min_track_frames=config.MIN_TRACK_FRAMES, max_history=50):
        self.detector = detector
        self.min_track_frames = min_track_frames
        self.max_history = max_history
        
        # Trajectory history: track_id -> deque of (center_x, center_y, frame_idx, class_id, conf)
        self.trajectories = defaultdict(lambda: deque(maxlen=self.max_history))
        # Total observation count per track ID for temporal consistency
        self.track_frame_counts = defaultdict(int)
        # Class assignment per track ID (using majority voting or highest confidence)
        self.track_classes = {}
        # Last seen frame index per track ID
        self.last_seen_frame = {}
        
    def update(self, frame, frame_idx, roi_polygon, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Executes YOLOv8 with BoT-SORT tracking, validates detections, and updates trajectories.
        Returns a list of verified active vehicle detections for the current frame.
        """
        # Run YOLOv8 BoT-SORT tracking
        results = self.detector.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=confidence_threshold,
            iou=config.IOU_THRESHOLD,
            device=self.detector.device,
            verbose=False
        )
        
        raw_boxes = results[0].boxes if (results and len(results) > 0) else None
        valid_detections = self.detector.filter_detections(
            raw_boxes,
            frame.shape,
            roi_polygon,
            confidence_threshold
        )
        
        active_vehicles = []
        
        for det in valid_detections:
            track_id = det["track_id"]
            if track_id is None:
                continue
                
            cx, cy = det["center"]
            cls_id = det["class_id"]
            conf = det["confidence"]
            
            # Update temporal observation count
            self.track_frame_counts[track_id] += 1
            self.last_seen_frame[track_id] = frame_idx
            
            # Store trajectory history point
            self.trajectories[track_id].append({
                "center_x": cx,
                "center_y": cy,
                "frame_number": frame_idx,
                "class_id": cls_id,
                "confidence": conf
            })
            
            # Keep consistent vehicle class
            if track_id not in self.track_classes:
                self.track_classes[track_id] = cls_id
            
            # Enforce temporal consistency: vehicle must persist for at least min_track_frames
            is_confirmed = self.track_frame_counts[track_id] >= self.min_track_frames
            
            det["confirmed"] = is_confirmed
            det["trajectory"] = list(self.trajectories[track_id])
            active_vehicles.append(det)
            
        return active_vehicles

    def get_trajectory(self, track_id):
        """Returns the recorded trajectory points for a given vehicle ID."""
        return list(self.trajectories.get(track_id, []))

    def prune_old_tracks(self, current_frame, max_age_frames=120):
        """Removes stale tracks that have left the scene."""
        stale_ids = [
            tid for tid, last_frame in self.last_seen_frame.items()
            if current_frame - last_frame > max_age_frames
        ]
        for tid in stale_ids:
            if tid in self.trajectories:
                del self.trajectories[tid]
            if tid in self.last_seen_frame:
                del self.last_seen_frame[tid]
