"""
Main entry point for Standalone Traffic Video Processing Application.
Provides a simple desktop GUI with FIRST SCREEN:
  Traffic Video Analyzer
  [ Choose Video ]
  [ Start Analysis ]
Supports Single Road and Intersection modes, copies videos into uploads/,
and executes detection, BoT-SORT tracking, turn classification, Auto-ARIMA forecasting,
adaptive signal timing, and video output.
"""
import os
import sys
import shutil
import argparse
import config
from video_processor import VideoProcessor

def show_first_screen():
    """
    Displays the FIRST SCREEN desktop GUI:
      Traffic Video Analyzer
      [ Choose Video ]
      Road Layout: Single Road | 2 Roads | 3 Roads | 4 Roads | Custom
      [ Start Analysis ]
    Returns (video_path, mode, road_count) or (None, None, None) if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        selection = {"video_path": None, "mode": "intersection", "road_count": 4}

        root = tk.Tk()
        root.title("Traffic Video Analyzer")
        root.geometry("520x420")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.eval('tk::PlaceWindow . center')

        bg_color    = "#1E2228"
        fg_color    = "#E0E5EC"
        accent_color = "#00D2FF"
        dim_color   = "#A0AAB8"
        root.configure(bg=bg_color)

        # ── Header ──────────────────────────────────────────────────────
        tk.Label(
            root, text="Traffic Video Analyzer",
            font=("Segoe UI", 16, "bold"),
            fg=accent_color, bg=bg_color, pady=14
        ).pack()

        # ── Choose Video ─────────────────────────────────────────────────
        video_var = tk.StringVar(value="No video selected")
        video_label = tk.Label(
            root, textvariable=video_var,
            font=("Segoe UI", 9), fg=dim_color, bg=bg_color, wraplength=460
        )

        def on_choose_video():
            fpath = filedialog.askopenfilename(
                title="Choose Traffic Video",
                filetypes=[
                    ("Traffic Videos", "*.mp4 *.avi *.mov *.mkv"),
                    ("MP4 Files",  "*.mp4"),
                    ("AVI Files",  "*.avi"),
                    ("MOV Files",  "*.mov"),
                    ("MKV Files",  "*.mkv"),
                    ("All Files",  "*.*"),
                ]
            )
            if fpath:
                selection["video_path"] = fpath
                video_var.set(os.path.basename(fpath))

        tk.Button(
            root, text="Choose Video", command=on_choose_video,
            font=("Segoe UI", 10, "bold"),
            bg="#2A303C", fg=fg_color,
            activebackground="#3A4252", activeforeground="#FFFFFF",
            relief="groove", padx=15, pady=6, cursor="hand2"
        ).pack(pady=5)
        video_label.pack(pady=2)

        # ── Separator ────────────────────────────────────────────────────
        tk.Frame(root, bg="#3A4050", height=1).pack(fill="x", padx=20, pady=8)

        # ── Road Layout Label ────────────────────────────────────────────
        tk.Label(
            root, text="Road Layout",
            font=("Segoe UI", 10, "bold"), fg=accent_color, bg=bg_color
        ).pack()

        # ── Road-count radio buttons ──────────────────────────────────────
        # road_choice: "1" | "2" | "3" | "4" | "custom"
        road_choice_var = tk.StringVar(value="4")

        btn_frame = tk.Frame(root, bg=bg_color)
        btn_frame.pack(pady=6)

        road_options = [
            ("Single Road", "1"),
            ("2 Roads",     "2"),
            ("3 Roads",     "3"),
            ("4 Roads",     "4"),
            ("Custom",      "custom"),
        ]
        for label_text, val in road_options:
            tk.Radiobutton(
                btn_frame, text=label_text,
                variable=road_choice_var, value=val,
                font=("Segoe UI", 9), fg=fg_color, bg=bg_color,
                selectcolor="#2A303C",
                activebackground=bg_color, activeforeground=fg_color
            ).pack(side="left", padx=8)

        # ── Custom road-count entry (visible only when "Custom" selected) ──
        custom_frame = tk.Frame(root, bg=bg_color)
        custom_var = tk.StringVar(value="5")

        custom_lbl = tk.Label(
            custom_frame, text="Number of roads:",
            font=("Segoe UI", 9), fg=dim_color, bg=bg_color
        )
        custom_lbl.pack(side="left", padx=(0, 6))

        custom_entry = tk.Entry(
            custom_frame, textvariable=custom_var,
            font=("Segoe UI", 10), width=5,
            bg="#2A303C", fg=fg_color, insertbackground=fg_color,
            relief="groove"
        )
        custom_entry.pack(side="left")

        def _update_custom_visibility(*_):
            if road_choice_var.get() == "custom":
                custom_frame.pack(pady=4)
            else:
                custom_frame.pack_forget()

        road_choice_var.trace_add("write", _update_custom_visibility)
        # Don't pack custom_frame yet; show only when "Custom" is selected.

        # ── Separator ────────────────────────────────────────────────────
        tk.Frame(root, bg="#3A4050", height=1).pack(fill="x", padx=20, pady=10)

        # ── Start Analysis ───────────────────────────────────────────────
        def on_start_analysis():
            if not selection["video_path"]:
                messagebox.showwarning("Warning", "Please choose a video file first.")
                return

            choice = road_choice_var.get()
            if choice == "1":
                selection["mode"]       = "single"
                selection["road_count"] = 1
            elif choice == "custom":
                try:
                    n = int(custom_var.get())
                    if n < 1:
                        raise ValueError
                except ValueError:
                    messagebox.showwarning("Warning", "Please enter a valid number of roads (≥ 1).")
                    return
                if n == 1:
                    selection["mode"]       = "single"
                    selection["road_count"] = 1
                else:
                    selection["mode"]       = "intersection"
                    selection["road_count"] = n
            else:
                n = int(choice)
                selection["mode"]       = "single" if n == 1 else "intersection"
                selection["road_count"] = n

            root.destroy()

        tk.Button(
            root, text="Start Analysis", command=on_start_analysis,
            font=("Segoe UI", 11, "bold"),
            bg="#007ACC", fg="#FFFFFF",
            activebackground="#0098FF", activeforeground="#FFFFFF",
            relief="raised", padx=25, pady=6, cursor="hand2"
        ).pack(pady=6)

        root.mainloop()
        return selection["video_path"], selection["mode"], selection["road_count"]
    except Exception as e:
        print(f"[Warning] Could not initialize graphical GUI: {e}")
        return None, None, None

def copy_to_uploads(video_path, uploads_dir=config.UPLOADS_FOLDER):
    """Copies the user's selected video into the uploads/ directory."""
    os.makedirs(uploads_dir, exist_ok=True)
    basename = os.path.basename(video_path)
    destination = os.path.join(uploads_dir, basename)
    
    # If not already located in uploads/, copy it
    if os.path.abspath(video_path) != os.path.abspath(destination):
        if getattr(config, "DEBUG", False):
            print(f"[Upload] Copying video into {destination} ...")
        shutil.copy2(video_path, destination)
    return destination

def print_terminal_summary(summary):
    """
    Prints the clean summary statistics directly in the terminal as specified.
    """
    mode     = summary.get("video_metadata", {}).get("mode", "intersection")
    counts   = summary.get("cumulative_vehicle_counts", {})
    turning  = summary.get("turning_patterns", {})
    forecast = summary.get("forecasting", {})
    signal   = summary.get("signal_timing", {})
    files    = summary.get("output_files", {})

    print("\n" + "=" * 45)
    print("Vehicle Counts\n")
    for class_name in ["Car", "Bus", "Truck", "Three Wheeler", "Two Wheeler", "LCV", "Bicycle"]:
        print(f"{class_name}: {counts.get(class_name, 0)}")

    if mode == "single":
        forecast_val = forecast.get("final_forecast")
        if forecast_val is not None:
            print(f"\nAuto-ARIMA Forecast\n\nNext Traffic: {forecast_val} vehicles")
        else:
            status_msg = forecast.get("status", "Preparing forecast...")
            print(f"\nAuto-ARIMA Forecast\n\n{status_msg}")

        green_time = signal.get("recommended_green_seconds", config.MIN_GREEN_LIGHT)
        print(f"\nAdaptive Green Signal\n\nGreen Time: {green_time} sec")
    else:
        print("\nTurning Counts\n")
        print(f"Left: {turning.get('left_turn', 0)}")
        print(f"Straight: {turning.get('straight', 0)}")
        print(f"Right: {turning.get('right_turn', 0)}")

        forecast_val = forecast.get("final_forecast")
        if forecast_val is not None:
            print(f"\nAuto-ARIMA Forecast\n\nNext Traffic: {forecast_val} vehicles")
        else:
            status_msg = forecast.get("status", "Preparing forecast...")
            print(f"\nAuto-ARIMA Forecast\n\n{status_msg}")

        road_signals = signal.get("road_signals", {})
        if road_signals:
            print("\nRoad Signals & Green Time")
            for rname, rinfo in road_signals.items():
                print(f"\n{rname}:")
                print(f"  Vehicles: {rinfo.get('vehicles', 0)}")
                print(f"  Green Time: {rinfo.get('green_seconds', config.MIN_GREEN_LIGHT)} sec")
                r_pred = rinfo.get('forecast')
                if r_pred is not None:
                    print(f"  Auto-ARIMA Forecast: Next Traffic: {r_pred} vehicles")

    print("\n" + "=" * 45)

    print("\nSaved Outputs:")
    for key, path in files.items():
        print(f"  {key:<20}: {path}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Traffic Video Analyzer - Standalone Traffic Monitoring"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        type=str,
        help="Path to input traffic video file (.mp4, .avi, .mov, .mkv). If omitted, displays first screen."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "intersection"],
        default=None,
        help="Road layout mode: 'single' for single-road or 'intersection' for multi-road intersection"
    )
    parser.add_argument(
        "--roads",
        type=int,
        default=None,
        help="Exact number of road approaches to analyse (1 = single, 2/3/4/N = intersection)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for YOLOv8 (default: {config.DEFAULT_CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames in video)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_FOLDER,
        help=f"Directory to save output files (default: {config.OUTPUT_FOLDER})"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live side-by-side preview window during processing (LEFT: video, RIGHT: info)"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive zone selector and use saved or fallback zones directly"
    )

    args = parser.parse_args()

    video_path  = args.video
    mode        = args.mode
    road_count  = args.roads
    show_preview = args.show

    # If no video passed via CLI, launch the FIRST SCREEN GUI
    if not video_path:
        result = show_first_screen()
        if result is None or result[0] is None:
            print("No video selected. Exiting.")
            sys.exit(0)
        video_path, mode, road_count = result
        show_preview = True  # GUI analysis defaults to live side-by-side preview

    # Reconcile --mode and --roads flags
    if road_count is not None and road_count == 1:
        mode = "single"
    elif road_count is not None and road_count > 1:
        mode = "intersection"
    elif mode == "single":
        road_count = 1
    elif mode == "intersection" and road_count is None:
        road_count = config.DEFAULT_ROAD_COUNT

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(config.MODEL_PATH):
        print(f"ERROR: Model weights not found at {config.MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    # Copy video into uploads/ directory as required
    uploaded_video_path = copy_to_uploads(video_path)

    processor = VideoProcessor(
        confidence_threshold=args.conf,
        output_dir=args.output_dir
    )

    if getattr(config, "DEBUG", False):
        print(f"[Config] Mode: {mode or 'intersection'} | Roads: {road_count or config.DEFAULT_ROAD_COUNT}")

    try:
        summary = processor.process_video(
            video_path=uploaded_video_path,
            max_frames=args.frames,
            show=show_preview,
            mode=mode,
            road_count=road_count,
            interactive_zones=not args.no_interactive
        )
        print_terminal_summary(summary)
    except Exception as e:
        if "Zone selection was cancelled" in str(e):
            print("\nAnalysis cancelled: Zone definition was cancelled by user.")
            sys.exit(0)
        print(f"\nERROR during video processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


