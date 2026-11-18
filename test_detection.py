"""
Independent test script for validating YOLOv8 detection and BoT-SORT tracking.
Loads best.pt, opens a video file, runs frame-by-frame inference, and prints/verifies:
- Bounding boxes [x1, y1, x2, y2]
- Vehicle classes
- Confidence scores
- BoT-SORT tracking IDs
"""
import os
import sys
import argparse
import cv2
from ultralytics import YOLO
import config

def run_test_detection(video_path=None, max_frames=60, confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD):
    print("=" * 65)
    print("TESTING YOLOv8 DETECTION AND BoT-SORT TRACKING PIPELINE")
    print("=" * 65)
    
    if not video_path:
        video_path = os.path.join(config.BASE_DIR, "sample_videos", "real_traffic.mp4")
        
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at: {video_path}")
        return False
        
    if not os.path.exists(config.MODEL_PATH):
        print(f"ERROR: Model file not found at: {config.MODEL_PATH}")
        return False

    print(f"Loading Model: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)
    
    print(f"Opening Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")
    print(f"Testing on first {max_frames} frames with confidence threshold: {confidence_threshold:.2f}")
    print("-" * 65)
    
    frame_idx = 0
    total_detections_logged = 0
    tracked_ids_seen = set()
    sample_frame_saved = False
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Run YOLOv8 detection with BoT-SORT tracker
        results = model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=confidence_threshold,
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            verbose=False
        )
        
        frame_detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = config.DISPLAY_NAMES.get(cls_id, f"Class_{cls_id}")
                
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())
                    tracked_ids_seen.add(track_id)
                    
                frame_detections.append({
                    "track_id": track_id,
                    "class": class_name,
                    "conf": conf,
                    "bbox": coords
                })
                total_detections_logged += 1
                
                # Annotate sample frame
                if not sample_frame_saved and frame_idx >= 15:
                    label = f"ID: {track_id} {class_name} {conf:.2f}" if track_id is not None else f"{class_name} {conf:.2f}"
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (coords[0], max(coords[1] - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        # Log periodically
        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"Frame {frame_idx:03d}/{max_frames}: {len(frame_detections)} vehicles detected | Total Unique Track IDs so far: {len(tracked_ids_seen)}")
            for det in frame_detections[:3]:
                tid_str = f"ID: {det['track_id']}" if det['track_id'] is not None else "ID: None"
                print(f"    -> {tid_str} | {det['class']} | Conf: {det['conf']:.2f} | BBox: {det['bbox'].tolist()}")
                
        # Save sample annotated frame
        if not sample_frame_saved and frame_idx >= 15 and len(frame_detections) > 0:
            sample_path = os.path.join(config.OUTPUT_FOLDER, "sample_detection_test.jpg")
            cv2.imwrite(sample_path, frame)
            sample_frame_saved = True
            print(f"    [Sample annotated frame saved to: {sample_path}]")
            
    cap.release()
    print("-" * 65)
    print(f"TEST SUMMARY:")
    print(f"Frames Processed: {frame_idx}")
    print(f"Total Detections: {total_detections_logged}")
    print(f"Total Unique Track IDs: {len(tracked_ids_seen)}")
    print(f"Unique Track IDs: {sorted(list(tracked_ids_seen))[:20]} ...")
    print("=" * 65)
    
    if len(tracked_ids_seen) > 0:
        print("SUCCESS: Real detection and BoT-SORT tracking verified!")
        return True
    else:
        print("WARNING: No tracked vehicles found. Check confidence threshold or video content.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8 + BoT-SORT on Video")
    parser.add_argument("--video", type=str, default=None, help="Path to traffic video file")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to test")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()
    
    run_test_detection(args.video, args.frames, args.conf)
