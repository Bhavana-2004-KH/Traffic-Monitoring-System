"""
Verification script for testing the complete end-to-end processing pipeline.
Runs video_processor on sample video and checks all stage outputs.
"""
import os
import sys
import json
import cv2
import config
from video_processor import VideoProcessor

def test_full_pipeline(video_path=None, test_frames=180):
    print("=" * 70)
    print("TESTING FULL END-TO-END PIPELINE")
    print("=" * 70)
    
    if not video_path:
        video_path = os.path.join(config.BASE_DIR, "sample_videos", "real_traffic.mp4")
        
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return False
        
    processor = VideoProcessor(confidence_threshold=0.50)
    print(f"Loading video: {video_path}")
    processor.open_video(video_path, interactive_zones=False)
    
    print(f"Processing first {test_frames} frames to test tracking, counting, ARIMA, and signal timing...")
    
    for i in range(test_frames):
        ret, frame, stats = processor.process_frame()
        if not ret:
            print(f"Video ended early at frame {i}")
            break
            
        if (i + 1) % 30 == 0 or (i + 1) == test_frames:
            print(f"Frame {i+1:03d}/{test_frames} | "
                  f"Active: {stats['current_vehicles']} | "
                  f"Total Counted: {stats['total_counted']} | "
                  f"Straight: {stats['turning_counts']['straight']}, "
                  f"Left: {stats['turning_counts']['left']}, "
                  f"Right: {stats['turning_counts']['right']} | "
                  f"{stats['forecast_display']} | "
                  f"{stats['green_text']}")
                  
            # Save sample annotated frame for visual inspection
            if i + 1 == 90 or i + 1 == test_frames:
                annotated_sample = os.path.join(config.OUTPUT_FOLDER, f"sample_annotated_frame_{i+1}.jpg")
                cv2.imwrite(annotated_sample, frame)
                print(f"  -> Saved sample annotated frame: {annotated_sample}")

                
    summary = processor.finalize()
    
    print("\n" + "=" * 70)
    print("CHECKING GENERATED OUTPUT FILES:")
    print("=" * 70)
    
    expected_files = [
        processor.output_video_path,
        processor.output_counts_csv,
        processor.output_forecast_csv,
        processor.output_summary_json
    ]
    
    all_exist = True
    for fpath in expected_files:
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            print(f"[OK] {os.path.basename(fpath)} exists ({sz} bytes)")
        else:
            print(f"[FAIL] {os.path.basename(fpath)} does NOT exist!")
            all_exist = False
            
    # Print output.json summary
    if os.path.exists(processor.output_summary_json):
        print("\nSummary JSON:")
        with open(processor.output_summary_json, "r") as f:
            print(json.dumps(json.load(f), indent=2))
            
    print("=" * 70)
    if all_exist:
        print("SUCCESS: Full pipeline verified end-to-end!")
        return True
    else:
        print("FAILURE: Some outputs were not generated.")
        return False

if __name__ == "__main__":
    test_full_pipeline()
