"""
Turning pattern detector module.
Detects vehicle turning behaviors (Straight, Left Turn, Right Turn)
using dynamic intersection entry/exit zones and trajectory displacement analysis.
Supports arbitrary intersection geometries (3-way, 4-way, 5-way, etc.).
"""
import math
import numpy as np
from shapely.geometry import Point, Polygon
import config

class TurningDetector:
    def __init__(self, zones_pixel=None, zone_polygons=None, mode="intersection"):
        self.mode = mode.lower() if mode else "intersection"
        self.zones = zones_pixel or {}
        self.zone_polygons = zone_polygons or {}
        
        # Intersection center and road centroids computed dynamically
        self.road_centroids = {}
        self.intersection_center = None
        if self.zones and self.mode == "intersection":
            self._compute_geometry()
            
        # Vehicle movement tracking
        # track_id -> {"entry_zone": str, "entry_point": (x,y), "exit_zone": str, "counted": bool, "movement": str}
        self.vehicle_states = {}
        
        # Active vehicle counts per road for signal allocation
        self.active_road_counts = {}
        
        # Cumulative turning counters
        self.straight_count = 0
        self.left_count = 0
        self.right_count = 0

    def set_zones(self, zones_pixel, zone_polygons=None, mode=None):
        """Configures dynamic road zones and operating mode for the current video."""
        if mode:
            self.mode = mode.lower()
        self.zones = zones_pixel or {}
        if zone_polygons is not None:
            self.zone_polygons = zone_polygons
        else:
            self.zone_polygons = {name: Polygon(pts) for name, pts in self.zones.items() if len(pts) >= 3}
        if self.mode == "intersection":
            self._compute_geometry()


    def _compute_geometry(self):
        """Computes road zone centroids and overall intersection center point."""
        self.road_centroids = {}
        cx_list, cy_list = [], []
        
        for name, pts in self.zones.items():
            pts_arr = np.array(pts, dtype=np.float32)
            c_x = float(np.mean(pts_arr[:, 0]))
            c_y = float(np.mean(pts_arr[:, 1]))
            self.road_centroids[name] = (c_x, c_y)
            cx_list.append(c_x)
            cy_list.append(c_y)
            
        if cx_list and cy_list:
            self.intersection_center = (float(np.mean(cx_list)), float(np.mean(cy_list)))
        else:
            self.intersection_center = None

    def identify_zone(self, point):
        """Returns the road zone name containing the point, or None."""
        p = Point(point[0], point[1])
        for zone_name, poly in self.zone_polygons.items():
            if poly.contains(p):
                return zone_name
        return None

    def classify_turn_from_geometry(self, entry_zone, exit_zone):
        """
        Classifies turn (Straight, Left, Right) based on entry and exit road geometry.
        Works for any 3-way, 4-way, or N-way intersection layout.
        """
        if not self.intersection_center or entry_zone not in self.road_centroids or exit_zone not in self.road_centroids:
            return "Straight"
            
        if entry_zone == exit_zone:
            return None
            
        c_entry = self.road_centroids[entry_zone]
        c_exit = self.road_centroids[exit_zone]
        c_center = self.intersection_center
        
        # Vector entering the intersection from entry road towards center
        v_in = np.array([c_center[0] - c_entry[0], c_center[1] - c_entry[1]], dtype=np.float64)
        # Vector exiting the intersection from center towards exit road
        v_out = np.array([c_exit[0] - c_center[0], c_exit[1] - c_center[1]], dtype=np.float64)
        
        len_in = np.linalg.norm(v_in)
        len_out = np.linalg.norm(v_out)
        if len_in < 1e-3 or len_out < 1e-3:
            return "Straight"
            
        theta_in = math.atan2(v_in[1], v_in[0])
        theta_out = math.atan2(v_out[1], v_out[0])
        
        # Deflection angle from incoming to outgoing heading
        diff_deg = math.degrees(theta_out - theta_in)
        while diff_deg > 180:
            diff_deg -= 360
        while diff_deg < -180:
            diff_deg += 360
            
        # In screen coordinates (y-axis points downwards):
        # Straight: approximately aligned with incoming heading (|diff| <= 40 deg)
        # Positive deflection: Clockwise rotation -> Right Turn
        # Negative deflection: Counter-clockwise rotation -> Left Turn
        if abs(diff_deg) <= 40:
            return "Straight"
        elif diff_deg > 40:
            return "Right"
        else:
            return "Left"

    def analyze_trajectory_deflection(self, trajectory):
        """
        Kinematic trajectory displacement analysis:
        Computes angular deflection between initial travel vector and final travel vector.
        """
        if len(trajectory) < 15:
            return None
            
        start_pt = np.array([trajectory[0]["center_x"], trajectory[0]["center_y"]], dtype=np.float64)
        mid_pt = np.array([trajectory[len(trajectory)//2]["center_x"], trajectory[len(trajectory)//2]["center_y"]], dtype=np.float64)
        end_pt = np.array([trajectory[-1]["center_x"], trajectory[-1]["center_y"]], dtype=np.float64)
        
        v1 = mid_pt - start_pt
        v2 = end_pt - mid_pt
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        if len_v1 < 20 or len_v2 < 20:
            return None
            
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        diff_deg = math.degrees(angle2 - angle1)
        
        while diff_deg > 180:
            diff_deg -= 360
        while diff_deg < -180:
            diff_deg += 360
            
        if abs(diff_deg) <= 22:
            return "Straight"
        elif diff_deg > 22:
            return "Right"
        else:
            return "Left"

    def update(self, active_vehicles, frame_width=None, frame_height=None):
        """
        Updates turning detection and per-road vehicle counts.
        In single-road mode, turning is Not applicable.
        In intersection mode, enforces multi-frame trajectory before classifying turn.
        """
        # 1. Single Road Mode Handling
        if self.mode == "single":
            self.active_road_counts = {"Single Road": len(active_vehicles)}
            return {
                "straight": "Not applicable",
                "left": "Not applicable",
                "right": "Not applicable"
            }
            
        # 2. Intersection Mode: Track active vehicle distribution per road
        self.active_road_counts = {name: 0 for name in self.zones.keys()}
        
        for vehicle in active_vehicles:
            track_id = vehicle["track_id"]
            if not vehicle.get("confirmed", False):
                continue
                
            center = vehicle["center"]
            trajectory = vehicle.get("trajectory", [])
            
            # Map current road location
            current_zone = self.identify_zone(center)
            if current_zone and current_zone in self.active_road_counts:
                self.active_road_counts[current_zone] += 1
            
            if track_id not in self.vehicle_states:
                self.vehicle_states[track_id] = {
                    "entry_zone": current_zone,
                    "entry_point": center,
                    "exit_zone": None,
                    "counted": False,
                    "movement": None
                }
            else:
                state = self.vehicle_states[track_id]
                if state["counted"]:
                    continue
                    
                # If entry zone was not captured on initial detection, record it when vehicle touches first zone
                if state["entry_zone"] is None and current_zone is not None:
                    state["entry_zone"] = current_zone
                    state["entry_point"] = center
                elif state["entry_zone"] is not None and current_zone is not None and current_zone != state["entry_zone"]:
                    # Vehicle transitioned from entry road to exit road
                    # Enforce multi-frame trajectory: require at least 8 trajectory points to verify movement
                    if len(trajectory) >= 8:
                        state["exit_zone"] = current_zone
                        movement = self.classify_turn_from_geometry(state["entry_zone"], state["exit_zone"])
                        if movement:
                            self._register_turn(track_id, movement)
                            continue
                        
                # Trajectory analysis fallback if vehicle traversed intersection without explicit zone hit
                if not state["counted"] and len(trajectory) >= 30:
                    movement = self.analyze_trajectory_deflection(trajectory)
                    if movement:
                        self._register_turn(track_id, movement)
                        
        return {
            "straight": self.straight_count,
            "left": self.left_count,
            "right": self.right_count
        }

    def _register_turn(self, track_id, movement):
        """Registers the turning event once per vehicle ID."""
        state = self.vehicle_states[track_id]
        if state["counted"]:
            return
            
        state["counted"] = True
        state["movement"] = movement
        
        if movement == "Straight":
            self.straight_count += 1
        elif movement == "Left":
            self.left_count += 1
        elif movement == "Right":
            self.right_count += 1

    def get_movement(self, track_id):
        """Returns the detected movement classification for a vehicle (Straight, Left, Right)."""
        if self.mode == "single":
            return None
        if track_id in self.vehicle_states:
            return self.vehicle_states[track_id].get("movement")
        return None


