"""
Detector module for vehicle detection using YOLOv8.
Handles model loading, inference, confidence filtering, and ROI spatial validation.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import config

class VehicleDetector:
    def __init__(self, model_path=config.MODEL_PATH, device=config.DEVICE):
        self.model_path = model_path
        self.device = device
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
            
        self.model = YOLO(self.model_path)
        self.class_names = config.VEHICLE_CLASSES
        self.display_names = config.DISPLAY_NAMES
        
    def get_roi_polygon(self, frame_width, frame_height, roi_normalized=None):
        """
        Converts normalized ROI coordinates into pixel coordinates for the given frame size.
        """
        if roi_normalized is None:
            roi_normalized = config.DEFAULT_ROI_NORMALIZED
            
        pts = []
        for norm_x, norm_y in roi_normalized:
            px = int(norm_x * frame_width)
            py = int(norm_y * frame_height)
            pts.append((px, py))
            
        return Polygon(pts), np.array(pts, dtype=np.int32)

    def is_inside_roi(self, center_point, roi_polygon):
        """
        Verifies if the vehicle center (or contact patch) is within the valid road ROI.
        """
        if roi_polygon is None:
            return True
        return roi_polygon.contains(Point(center_point[0], center_point[1]))

    def filter_detections(self, raw_boxes, frame_shape, roi_polygon, confidence_threshold):
        """
        Applies confidence filtering, bounding box size validation, and ROI polygon filtering.
        """
        valid_detections = []
        if raw_boxes is None or len(raw_boxes) == 0:
            return valid_detections
            
        for box in raw_boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < confidence_threshold:
                continue
                
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id not in self.class_names:
                continue
                
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            width = x2 - x1
            height = y2 - y1
            if width < config.MIN_BBOX_WIDTH or height < config.MIN_BBOX_HEIGHT:
                continue
                
            # Use lower center of the bounding box (vehicle road contact area)
            center_x = int((x1 + x2) / 2.0)
            center_y = int(y2 - (height * 0.2))
            
            if roi_polygon is not None and not self.is_inside_roi((center_x, center_y), roi_polygon):
                continue
                
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
                
            valid_detections.append({
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": self.display_names.get(cls_id, f"Class_{cls_id}"),
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y)
            })
            
        return valid_detections
