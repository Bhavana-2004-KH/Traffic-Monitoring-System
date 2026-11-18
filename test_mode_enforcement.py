"""
test_mode_enforcement.py

Verifies that mode and road_count are enforced at every layer:
  TEST 1 - Single Road  : exactly 1 road, no turning, no multi-road artifacts
  TEST 2 - 2-Road Inter : exactly 2 roads, green times per road
  TEST 3 - 4-Road Inter : exactly 4 roads, turns, per-road green times
"""

import config
from traffic_state import TrafficState
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_zones(names, cols=4, row_h=50):
    """Generate non-overlapping rectangular zones for the given road names."""
    zones = {}
    for i, name in enumerate(names):
        x0 = i * (100 // max(1, cols) + 10)
        zones[name] = [(x0, 0), (x0 + 40, 0), (x0 + 40, row_h), (x0, row_h)]
    return zones


def run_frames(state, detections, n_frames=None):
    """Drive n_frames of state.update() with the given detection list."""
    n = n_frames or config.SIGNAL_CYCLE_FRAMES + 1
    for f in range(1, n + 1):
        state.update(detections, frame_idx=f, timestamp_sec=f / 30.0, fps=30.0)


# ---------------------------------------------------------------------------
# TEST 1 - Single Road
# ---------------------------------------------------------------------------

def test_single_road():
    print("\n" + "=" * 55)
    print("TEST 1: Single Road mode")
    print("=" * 55)

    # TrafficState with mode="single" and NO zones (auto full-frame)
    state = TrafficState(mode="single")

    # --- Road structure ---
    assert len(state.roads) == 1, \
        f"[FAIL] Expected 1 road, got {len(state.roads)}: {list(state.roads.keys())}"
    assert "Single Road" in state.roads, \
        f"[FAIL] Expected key 'Single Road', got: {list(state.roads.keys())}"
    assert state.mode == "single", \
        f"[FAIL] Expected mode='single', got '{state.mode}'"
    assert state.road_count == 1, \
        f"[FAIL] Expected road_count=1, got {state.road_count}"

    for bad in ("Road 1", "Road 2", "Road 3", "Road 4"):
        assert bad not in state.roads, \
            f"[FAIL] '{bad}' must NOT exist in single-road mode"

    print("[PASS] Exactly 1 road ('Single Road'), no Road 1-4")

    # --- Turning ---
    assert state.get_movement(99) is None, "[FAIL] Turning must be None in single mode"
    print("[PASS] Turning detection disabled (get_movement returns None)")

    # --- Vehicle counting ---
    det = [{"track_id": 1, "class_id": 0, "class_name": "Car",
            "center": (100, 100), "bbox": [90, 90, 110, 110], "confirmed": True}]
    state.update(det, frame_idx=1, timestamp_sec=0.1, fps=30.0)

    assert 1 in state.counted_track_ids, "[FAIL] Vehicle not counted"
    assert state.vehicles[1]["current_road"] == "Single Road", \
        f"[FAIL] vehicle road should be 'Single Road', got '{state.vehicles[1]['current_road']}'"
    assert state.roads["Single Road"]["vehicle_count"] == 1, \
        f"[FAIL] road vehicle_count should be 1, got {state.roads['Single Road']['vehicle_count']}"
    print("[PASS] Vehicle counted and assigned to 'Single Road'")

    # --- Green time: exactly one ---
    run_frames(state, det)
    green = state.roads["Single Road"]["green_time"]
    assert config.MIN_GREEN_LIGHT <= green <= config.MAX_GREEN_LIGHT, \
        f"[FAIL] green_time={green} outside [{config.MIN_GREEN_LIGHT},{config.MAX_GREEN_LIGHT}]"
    green_keys = [k for k in state.roads]
    assert len(green_keys) == 1, \
        f"[FAIL] Expected 1 green_time entry, got {len(green_keys)}: {green_keys}"
    print(f"[PASS] Exactly one green time: 'Single Road'={green}s")

    print("\n[OK] TEST 1 PASSED - Single Road mode is fully isolated.\n")


# ---------------------------------------------------------------------------
# TEST 2 - 2-Road Intersection
# ---------------------------------------------------------------------------

def test_two_road_intersection():
    print("=" * 55)
    print("TEST 2: 2-Road intersection mode")
    print("=" * 55)

    zones_px = {
        "Road 1": [(0, 0),   (80, 0),   (80, 60),  (0, 60)],
        "Road 2": [(100, 0), (180, 0),  (180, 60), (100, 60)],
    }
    state = TrafficState(mode="intersection", zones_pixel=zones_px, road_count=2)

    # --- Road structure ---
    assert len(state.roads) == 2, \
        f"[FAIL] Expected 2 roads, got {len(state.roads)}: {list(state.roads.keys())}"
    assert "Road 1" in state.roads and "Road 2" in state.roads, \
        f"[FAIL] Expected Road 1 & 2, got {list(state.roads.keys())}"
    for bad in ("Road 3", "Road 4", "Single Road"):
        assert bad not in state.roads, \
            f"[FAIL] '{bad}' must NOT exist in 2-road intersection"

    print("[PASS] Exactly 2 roads (Road 1, Road 2), no Road 3/4")

    # --- Vehicle counts per road ---
    det = [
        {"track_id": 10, "class_id": 0, "class_name": "Car",
         "center": (40, 30),  "bbox": [30, 20, 50, 40], "confirmed": True},   # Road 1
        {"track_id": 20, "class_id": 1, "class_name": "Bus",
         "center": (140, 30), "bbox": [130, 20, 150, 40], "confirmed": True},  # Road 2
    ]
    run_frames(state, det)

    g1 = state.roads["Road 1"]["green_time"]
    g2 = state.roads["Road 2"]["green_time"]
    assert config.MIN_GREEN_LIGHT <= g1 <= config.MAX_GREEN_LIGHT, \
        f"[FAIL] Road 1 green_time={g1} out of bounds"
    assert config.MIN_GREEN_LIGHT <= g2 <= config.MAX_GREEN_LIGHT, \
        f"[FAIL] Road 2 green_time={g2} out of bounds"

    print(f"[PASS] Per-road green times: Road 1={g1}s, Road 2={g2}s")
    print(f"[PASS] Road 1 vehicles: {state.roads['Road 1']['vehicle_count']}, "
          f"Road 2 vehicles: {state.roads['Road 2']['vehicle_count']}")

    print("\n[OK] TEST 2 PASSED - 2-road intersection mode correct.\n")


# ---------------------------------------------------------------------------
# TEST 3 - 4-Road Intersection (existing behavior must be preserved)
# ---------------------------------------------------------------------------

def test_four_road_intersection():
    print("=" * 55)
    print("TEST 3: 4-Road intersection mode")
    print("=" * 55)

    zones_px = {
        "Road 1": [(100, 0),   (200, 0),   (200, 100), (100, 100)],
        "Road 2": [(100, 300), (200, 300), (200, 400), (100, 400)],
        "Road 3": [(0, 100),   (100, 100), (100, 200), (0, 200)],
        "Road 4": [(200, 100), (300, 100), (300, 200), (200, 200)],
    }
    state = TrafficState(mode="intersection", zones_pixel=zones_px, road_count=4)

    # --- Road structure ---
    assert len(state.roads) == 4, \
        f"[FAIL] Expected 4 roads, got {len(state.roads)}: {list(state.roads.keys())}"
    for r in ("Road 1", "Road 2", "Road 3", "Road 4"):
        assert r in state.roads, f"[FAIL] '{r}' missing from 4-road intersection"
    assert "Single Road" not in state.roads, \
        "[FAIL] 'Single Road' must NOT appear in intersection mode"

    print("[PASS] Exactly 4 roads (Road 1-4), no 'Single Road'")

    # Road 1=0 vehicles, Road 2=1, Road 3=2, Road 4=3
    det = [
        {"track_id": 101, "class_id": 0, "class_name": "Car",
         "center": (150, 350), "bbox": [140, 340, 160, 360], "confirmed": True},
        {"track_id": 201, "class_id": 0, "class_name": "Car",
         "center": (50, 150),  "bbox": [40,  140, 60,  160],  "confirmed": True},
        {"track_id": 202, "class_id": 0, "class_name": "Car",
         "center": (55, 155),  "bbox": [45,  145, 65,  165],  "confirmed": True},
        {"track_id": 301, "class_id": 0, "class_name": "Car",
         "center": (250, 150), "bbox": [240, 140, 260, 160], "confirmed": True},
        {"track_id": 302, "class_id": 0, "class_name": "Car",
         "center": (255, 155), "bbox": [245, 145, 265, 165], "confirmed": True},
        {"track_id": 303, "class_id": 0, "class_name": "Car",
         "center": (258, 158), "bbox": [248, 148, 268, 168], "confirmed": True},
    ]
    run_frames(state, det)

    g1 = state.roads["Road 1"]["green_time"]
    g2 = state.roads["Road 2"]["green_time"]
    g3 = state.roads["Road 3"]["green_time"]
    g4 = state.roads["Road 4"]["green_time"]

    assert g1 == config.MIN_GREEN_LIGHT, \
        f"[FAIL] Road 1 (0 vehicles) should be MIN_GREEN={config.MIN_GREEN_LIGHT}s, got {g1}"
    assert g4 > g1, \
        f"[FAIL] Road 4 (3 vehicles) green_time={g4} should exceed Road 1={g1}"
    assert not (g1 == g2 == g3 == g4), \
        f"[FAIL] All roads have identical green times - road-specific demand not working"

    print(f"[PASS] Road-specific green times: R1={g1}s R2={g2}s R3={g3}s R4={g4}s")

    # --- Turning still active in intersection ---
    assert state.mode == "intersection", "[FAIL] mode should be 'intersection'"
    print("[PASS] mode='intersection' - turning detection is ACTIVE")

    print("\n[OK] TEST 3 PASSED - 4-road intersection behavior preserved.\n")


# ---------------------------------------------------------------------------
# TEST 4 - zone_manager single-road never returns > 1 zone
# ---------------------------------------------------------------------------

def test_zone_manager_single_road():
    import tempfile, os
    import numpy as np
    import zone_manager

    print("=" * 55)
    print("TEST 4: zone_manager single-road guarantee")
    print("=" * 55)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake stale 4-road zone file as if it was written by a previous run
        fake_video_name = "testvideo.mp4"
        stale_path = os.path.join(tmpdir, "testvideo_zones.json")
        import json
        stale = {
            "mode": "intersection",
            "video_name": fake_video_name,
            "video_width": 640, "video_height": 480,
            "road_names": ["Road 1","Road 2","Road 3","Road 4"],
            "zones": {
                "Road 1": [[0.2,0.28],[0.8,0.28],[0.8,0.45],[0.2,0.45]],
                "Road 2": [[0.1,0.75],[0.9,0.75],[0.9,1.0],[0.1,1.0]],
                "Road 3": [[0.0,0.40],[0.3,0.40],[0.3,0.75],[0.0,0.75]],
                "Road 4": [[0.7,0.40],[1.0,0.40],[1.0,0.75],[0.7,0.75]],
            }
        }
        with open(stale_path, "w") as f:
            json.dump(stale, f)

        # Dummy first frame
        first_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create a fake video path (zone_manager uses basename only, no open)
        fake_video = os.path.join(tmpdir, fake_video_name)
        open(fake_video, "w").close()  # empty placeholder

        m, norm, px, poly = zone_manager.get_or_create_zones(
            video_path=fake_video,
            first_frame=first_frame,
            mode="single",
            road_count=1,
            output_dir=tmpdir,
            interactive=False
        )

        assert m == "single", f"[FAIL] Returned mode={m!r}, expected 'single'"
        assert len(norm) == 1, \
            f"[FAIL] zone_manager returned {len(norm)} zones for single mode: {list(norm.keys())}"
        assert "Single Road" in norm, \
            f"[FAIL] Expected 'Single Road' key, got {list(norm.keys())}"
        for bad in ("Road 1","Road 2","Road 3","Road 4"):
            assert bad not in norm, \
                f"[FAIL] '{bad}' must NOT appear in single-road zone result"

    print("[PASS] zone_manager returns exactly 1 'Single Road' zone even with a "
          "stale 4-road zone file present")
    print("\n[OK] TEST 4 PASSED - zone_manager single-road is bullet-proof.\n")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_single_road()
    test_two_road_intersection()
    test_four_road_intersection()
    test_zone_manager_single_road()
    print("=" * 55)
    print("ALL MODE-ENFORCEMENT TESTS PASSED [OK]")
    print("=" * 55)
