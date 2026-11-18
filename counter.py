"""
Counter module for tracking unique vehicles and class-specific tallies.
Prevents duplicate counts across frames by binding to BoT-SORT track IDs.
"""
from collections import defaultdict
import config

class VehicleCounter:
    def __init__(self):
        # Set of unique track IDs counted cumulatively
        self.counted_track_ids = set()
        
        # Cumulative counts per vehicle class
        self.class_counts = {cls_id: 0 for cls_id in config.VEHICLE_CLASSES.keys()}
        
        # History log for reporting and time-series extraction
        # Each entry: {"frame": int, "current_vehicles": int, "total_counted": int, "class_counts": dict}
        self.history = []
        
    def update(self, active_vehicles, frame_idx):
        """
        Updates cumulative unique counts using confirmed active vehicles.
        Returns a dictionary summarizing current and cumulative statistics.
        """
        current_active_count = 0
        current_by_class = {cls_id: 0 for cls_id in config.VEHICLE_CLASSES.keys()}
        
        for vehicle in active_vehicles:
            # Only count vehicles that pass temporal consistency
            if not vehicle.get("confirmed", False):
                continue
                
            current_active_count += 1
            cls_id = vehicle["class_id"]
            current_by_class[cls_id] = current_by_class.get(cls_id, 0) + 1
            
            track_id = vehicle["track_id"]
            if track_id not in self.counted_track_ids:
                self.counted_track_ids.add(track_id)
                self.class_counts[cls_id] = self.class_counts.get(cls_id, 0) + 1
                
        snapshot = {
            "frame": frame_idx,
            "current_vehicles": current_active_count,
            "total_counted": len(self.counted_track_ids),
            "class_counts": {config.DISPLAY_NAMES[k]: v for k, v in self.class_counts.items()},
            "current_by_class": {config.DISPLAY_NAMES[k]: v for k, v in current_by_class.items()}
        }
        self.history.append(snapshot)
        return snapshot

    def get_summary(self):
        """Returns the current cumulative tally."""
        return {
            "total_counted": len(self.counted_track_ids),
            "class_counts": {config.DISPLAY_NAMES[k]: v for k, v in self.class_counts.items()}
        }
