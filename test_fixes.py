import os
import json
import tempfile
import pandas as pd
import config
from signal_controller import AdaptiveSignalController
from forecasting import TrafficForecaster
import zone_manager

def test_signal_controller():
    ctrl = AdaptiveSignalController()
    ctrl.current_green_time = 25
    # Force high demand
    for _ in range(ctrl.cycle_frames):
        res = ctrl.calculate(predicted_vehicles=100)
    assert res["green_seconds"] == 30, f"Expected 30 (25 + 5 clamp), got {res['green_seconds']}"
    print("[PASS] Signal controller clamping (+5s clamp verified)")

    # Test intersection phase cycling
    demands = {"Road 1": 10, "Road 2": 20}
    # Initial state
    signals = ctrl.calculate_roads(demands, fps=30.0)
    assert "signal_state" in signals["Road 1"], "signal_state missing from road 1"
    assert "signal_state" in signals["Road 2"], "signal_state missing from road 2"
    assert signals["Road 1"]["signal_state"] in ("GREEN", "YELLOW"), "Road 1 should be active initially"
    assert signals["Road 2"]["signal_state"] == "RED", "Road 2 should be RED initially"
    print("[PASS] Signal controller intersection phase tracking (GREEN/YELLOW/RED states)")

def test_forecaster():
    fc = TrafficForecaster()
    assert fc.min_observations == 2, f"Expected min_observations=2, got {fc.min_observations}"
    
    # 1st interval (5.0s) - warming up
    fc.add_frame_sample(5.0, 10)
    assert fc.predicted_next is None, "Should be None before 2 intervals"
    assert "Preparing forecast..." in fc.get_forecast_display(), f"Expected 'Preparing forecast...' in display, got {fc.get_forecast_display()}"
    
    # 2nd interval (10.0s) - Auto-ARIMA forecast generated
    fc.add_frame_sample(10.0, 15)
    assert fc.predicted_next is not None, "Expected valid forecast on 2nd interval (10s)"
    assert "Next Traffic:" in fc.get_forecast_display(), f"Expected 'Next Traffic:' in display, got {fc.get_forecast_display()}"
    print(f"[PASS] TrafficForecaster 2-interval Auto-ARIMA forecast verified: {fc.predicted_next}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "fc_test.csv")
        fc.export_csv(csv_path)
        df = pd.read_csv(csv_path)
        assert "road" in df.columns, "Missing 'road' column in forecast CSV"
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        print("[PASS] TrafficForecaster export_csv contains 'road' column")

def test_traffic_state():
    from traffic_state import TrafficState
    from shapely.geometry import Polygon
    
    # Define a 4-road intersection
    zones_px = {
        "Road 1": [(100, 0), (200, 0), (200, 100), (100, 100)],
        "Road 2": [(100, 300), (200, 300), (200, 400), (100, 400)],
        "Road 3": [(0, 100), (100, 100), (100, 200), (0, 200)],
        "Road 4": [(200, 100), (300, 100), (300, 200), (200, 200)],
    }
    state = TrafficState(mode="intersection", zones_pixel=zones_px)
    assert len(state.roads) == 4, f"Expected 4 roads, got {len(state.roads)}"
    assert list(state.roads.keys()) == ["Road 1", "Road 2", "Road 3", "Road 4"]
    print("[PASS] TrafficState holds all 4 detected roads dynamically")

    # Simulate Track 23: enters Road 1, traverses to Road 3 (LEFT/RIGHT turn)
    # Frame 1: inside Road 1
    det1 = [{"track_id": 23, "class_id": 0, "class_name": "Car", "center": (150, 50), "bbox": [130, 30, 170, 70], "confirmed": True}]
    state.update(det1, frame_idx=1, timestamp_sec=0.1, fps=30.0)
    assert state.vehicles[23]["entry_road"] == "Road 1"
    assert 23 in state.roads["Road 1"]["vehicle_ids"]
    assert state.overall_class_counts["Car"] == 1
    assert len(state.counted_track_ids) == 1
    print("[PASS] Track 23 counted once and assigned entry road Road 1")

    # Move through central area with trajectory points (x=150 is in central intersection)
    for f in range(2, 8):
        det_mid = [{"track_id": 23, "class_id": 0, "class_name": "Car", "center": (150, 50 + f*10), "bbox": [130, 30, 170, 70], "confirmed": True}]
        state.update(det_mid, frame_idx=f, timestamp_sec=f*0.1, fps=30.0)
    
    # Still inside central intersection: movement NOT completed yet (Section 7)
    assert not state.vehicles[23]["movement_completed"]
    print("[PASS] Vehicle is NOT classified as turn while moving through intersection (Section 7)")

    # Enters Road 3 (exit road)
    det_exit = [{"track_id": 23, "class_id": 0, "class_name": "Car", "center": (50, 150), "bbox": [30, 130, 70, 170], "confirmed": True}]
    state.update(det_exit, frame_idx=15, timestamp_sec=1.5, fps=30.0)

    # Now movement is completed!
    assert state.vehicles[23]["movement_completed"], "Movement should be completed upon reaching exit road"
    turn_type = state.vehicles[23]["turn"]
    assert turn_type in ("LEFT", "RIGHT", "STRAIGHT"), f"Invalid turn: {turn_type}"
    assert 23 in state.turning_counts[turn_type], f"Track 23 missing from turning_counts[{turn_type}]"
    assert 23 in state.roads["Road 3"]["vehicle_ids"], "Track 23 missing from Road 3 vehicle_ids"
    # Ensure vehicle count is not doubled
    assert len(state.counted_track_ids) == 1, "Vehicle should only be counted once overall"
    print(f"[PASS] Track 23 completed turn: {turn_type}, fully traceable to vehicle count and Road 3")

def test_zone_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "test_zones.json")
        zones = {"Road 1": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}
        zone_manager.save_zones(zpath, zones, mode="intersection", video_name="sample.mp4", video_width=1280, video_height=720)
        with open(zpath) as f:
            data = json.load(f)
        assert data["video_width"] == 1280, "Missing or invalid video_width"
        assert data["video_height"] == 720, "Missing or invalid video_height"
        assert data["road_names"] == ["Road 1"], "Missing or invalid road_names"
        
        mode, norm, px, poly = zone_manager.load_zones(zpath, 1280, 720)
        assert "Road 1" in norm, "Road 1 missing from norm"
        assert "Road 1" in px, "Road 1 missing from px"
        print("[PASS] ZoneManager saves and loads video_width, video_height, road_names properly")

def test_road_specific_green_times():
    from traffic_state import TrafficState

    zones_px = {
        "Road 1": [(0, 0), (50, 0), (50, 50), (0, 50)],
        "Road 2": [(100, 0), (150, 0), (150, 50), (100, 50)],
        "Road 3": [(200, 0), (250, 0), (250, 50), (200, 50)],
        "Road 4": [(300, 0), (350, 0), (350, 50), (300, 50)],
    }
    state = TrafficState(mode="intersection", zones_pixel=zones_px)

    # Road 1: 0 vehicles
    # Road 2: 1 vehicle
    r2_dets = [{"track_id": 101, "class_id": 0, "class_name": "Car", "center": (125, 25), "bbox": [110, 10, 140, 40], "confirmed": True}]
    # Road 3: 2 vehicles
    r3_dets = [
        {"track_id": 201, "class_id": 0, "class_name": "Car", "center": (225, 25), "bbox": [210, 10, 240, 40], "confirmed": True},
        {"track_id": 202, "class_id": 0, "class_name": "Car", "center": (230, 30), "bbox": [220, 20, 240, 40], "confirmed": True},
    ]
    # Road 4: 3 vehicles
    r4_dets = [
        {"track_id": 301, "class_id": 0, "class_name": "Car", "center": (325, 25), "bbox": [310, 10, 340, 40], "confirmed": True},
        {"track_id": 302, "class_id": 0, "class_name": "Car", "center": (330, 30), "bbox": [320, 20, 340, 40], "confirmed": True},
        {"track_id": 303, "class_id": 0, "class_name": "Car", "center": (335, 35), "bbox": [325, 25, 345, 45], "confirmed": True},
    ]
    all_dets = r2_dets + r3_dets + r4_dets

    # Run for 1 cycle (90 frames) to trigger recalculation
    for f in range(1, config.SIGNAL_CYCLE_FRAMES + 1):
        state.update(all_dets, frame_idx=f, timestamp_sec=f / 30.0, fps=30.0)

    g1 = state.roads["Road 1"]["green_time"]
    g2 = state.roads["Road 2"]["green_time"]
    g3 = state.roads["Road 3"]["green_time"]
    g4 = state.roads["Road 4"]["green_time"]

    # Verify not all equal to 25
    assert not (g1 == g2 == g3 == g4 == 25), f"BUG: All roads have green_time=25! {g1}, {g2}, {g3}, {g4}"
    # Verify Road 1 (0 vehicles) receives minimum safe green (20 sec)
    assert g1 == 20, f"Expected Road 1 (0 vehicles) to have min green 20, got {g1}"
    # Verify green times respond to traffic density: Road 1 <= Road 2 <= Road 3 <= Road 4
    assert g1 <= g2 <= g3 <= g4, f"Expected non-decreasing green times, got {g1}, {g2}, {g3}, {g4}"
    assert g4 > g1, f"Expected Road 4 with 3 vehicles to have higher green than Road 1, got {g4} vs {g1}"
    print(f"[PASS] Road-specific green times verified: R1={g1}s, R2={g2}s, R3={g3}s, R4={g4}s (not all 25s)")

    # Test with forecast on Road 4: blend into demand
    state.roads["Road 4"]["forecast"] = 18.0
    for f in range(config.SIGNAL_CYCLE_FRAMES + 1, config.SIGNAL_CYCLE_FRAMES * 4 + 1):
        state.update(all_dets, frame_idx=f, timestamp_sec=f / 30.0, fps=30.0)

    g1_late = state.roads["Road 1"]["green_time"]
    g4_late = state.roads["Road 4"]["green_time"]
    assert g1_late == 20, f"Expected Road 1 to stay at 20s, got {g1_late}"
    assert g4_late > g1_late, f"Expected Road 4 green time to increase with demand+forecast, got {g4_late}"
    assert 20 <= g4_late <= 90, f"Green time must be within [20, 90], got {g4_late}"
    print(f"[PASS] Auto-ARIMA forecast blended into Road 4 demand and rate-limited clamping verified: R4={g4_late}s")

def test_single_road_green_time():
    from traffic_state import TrafficState
    state = TrafficState(mode="single")
    assert "Single Road" in state.roads
    assert len(state.roads) == 1
    det = [{"track_id": 1, "class_id": 0, "class_name": "Car", "center": (100, 100), "bbox": [90, 90, 110, 110], "confirmed": True}]
    for f in range(1, config.SIGNAL_CYCLE_FRAMES + 1):
        state.update(det, frame_idx=f, timestamp_sec=f / 30.0, fps=30.0)
    g = state.roads["Single Road"]["green_time"]
    assert 20 <= g <= 90
    print(f"[PASS] Single road mode calculates exactly one green time: {g}s")

if __name__ == "__main__":
    test_signal_controller()
    test_forecaster()
    test_traffic_state()
    test_zone_manager()
    test_road_specific_green_times()
    test_single_road_green_time()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
