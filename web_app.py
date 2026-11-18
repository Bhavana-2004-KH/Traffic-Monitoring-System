"""
web_app.py
Website backend server for Traffic Video Analyzer.
Provides a modern Flask dashboard connecting to the existing Traffic Video Analyzer.

STRICT RULE COMPLIANCE:
- Does NOT modify, overwrite, rename, or refactor any existing project file.
- Invokes the existing pipeline (main.py / video_processor.py) safely via subprocess.
- Reads and serves actual existing results, output files, and forecasts.
"""

import os
import sys
import time
import json
import re
import threading
import subprocess
from datetime import datetime
from urllib.parse import unquote
import cv2
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    abort
)
from werkzeug.utils import secure_filename

import config

# Setup directories from existing config
BASE_DIR = config.BASE_DIR
UPLOADS_DIR = config.UPLOADS_FOLDER
OUTPUTS_DIR = config.OUTPUT_FOLDER
SAMPLE_VIDEOS_DIR = os.path.join(BASE_DIR, "sample_videos")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_VIDEOS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB upload limit
app.config["JSON_SORT_KEYS"] = False

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def transcode_to_web_compatible(src_path, dst_path):
    """
    Transcodes an OpenCV-generated MP4 (mp4v codec) into an HTML5 browser-compatible
    MP4 (avc1 / H.264) using OpenCV and OpenH264.
    Runs fast (e.g. ~0.3s for 60-100 frames).
    """
    if not os.path.exists(src_path):
        return False
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            return False

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            writer.write(frame)

        cap.release()
        writer.release()
        return os.path.exists(dst_path) and os.path.getsize(dst_path) > 100
    except Exception as e:
        print(f"[Transcode] Error transcoding to web video: {e}")
        return False


# ---------------------------------------------------------------------------
# Background Job Manager
# ---------------------------------------------------------------------------

class JobManager:
    """Thread-safe background process coordinator for video analysis jobs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.worker_thread = None
        self.current_job = {
            "job_id": None,
            "status": "idle",  # "idle" | "processing" | "completed" | "failed" | "cancelled"
            "video_name": "",
            "video_stem": "",
            "video_path": "",
            "mode": "intersection",
            "roads": 4,
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 30.0,
            "active_vehicles": 0,
            "total_counted": 0,
            "green_text": "20s",
            "forecast_text": "Preparing forecast...",
            "turning_text": "",
            "live_roads": {},
            "start_time": 0.0,
            "elapsed_seconds": 0.0,
            "log_lines": [],
            "error": None,
            "summary": None,
            "output_video": None
        }

    def get_state(self):
        with self.lock:
            state = dict(self.current_job)
            if state["status"] == "processing" and state["start_time"] > 0:
                state["elapsed_seconds"] = round(time.time() - state["start_time"], 1)
            state["log_lines"] = state["log_lines"][-60:]
            return state

    def is_running(self):
        with self.lock:
            return self.current_job["status"] == "processing" and self.process is not None and self.process.poll() is None

    def cancel(self):
        with self.lock:
            if not (self.process and self.process.poll() is None):
                return False, "No analysis job is currently running."

            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        capture_output=True
                    )
                else:
                    self.process.terminate()
                    time.sleep(0.3)
                    if self.process.poll() is None:
                        self.process.kill()
            except Exception as e:
                print(f"[JobManager] Error terminating process: {e}")

            self.current_job["status"] = "cancelled"
            self.current_job["error"] = "Analysis was cancelled by user."
            self.current_job["log_lines"].append("Analysis cancelled by user.")
            return True, "Analysis job was cancelled."

    def start_job(self, video_path, mode="intersection", roads=4, max_frames=None):
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "An analysis job is already running. Please wait or cancel it first."

            video_basename = os.path.basename(video_path)
            video_stem = os.path.splitext(video_basename)[0]

            job_id = f"job_{int(time.time())}_{video_stem}"
            self.current_job = {
                "job_id": job_id,
                "status": "processing",
                "video_name": video_basename,
                "video_stem": video_stem,
                "video_path": video_path,
                "mode": mode,
                "roads": roads if mode == "intersection" else 1,
                "progress": 0,
                "current_frame": 0,
                "total_frames": 0,
                "fps": 30.0,
                "active_vehicles": 0,
                "total_counted": 0,
                "green_text": "20s",
                "forecast_text": "Preparing forecast...",
                "turning_text": "",
                "live_roads": {},
                "start_time": time.time(),
                "elapsed_seconds": 0.0,
                "log_lines": [f"Initiating analysis for {video_basename} (Mode: {mode}, Roads: {roads})"],
                "error": None,
                "summary": None,
                "output_video": None
            }

            self.worker_thread = threading.Thread(
                target=self._run_process,
                args=(video_path, mode, roads, max_frames, video_stem),
                daemon=True
            )
            self.worker_thread.start()
            return True, job_id

    def _run_process(self, video_path, mode, roads, max_frames, video_stem):
        cmd = [
            sys.executable,
            "-u",
            os.path.join(BASE_DIR, "main.py"),
            video_path,
            "--no-interactive",
            "--mode", mode,
            "--roads", str(roads)
        ]
        if max_frames:
            cmd.extend(["--frames", str(max_frames)])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
        except Exception as e:
            with self.lock:
                self.current_job["status"] = "failed"
                self.current_job["error"] = f"Failed to launch analysis process: {str(e)}"
            return

        re_frames = re.compile(r"Frames to process:\s*(\d+)")
        re_progress = re.compile(
            r"\[Processing\]\s+Frame\s+(\d+)/(\d+)\s*\(\s*(\d+)%\)\s*\|\s*Active:\s*(\d+)\s*\|\s*Total:\s*(\d+)\s*\|\s*(.*)"
        )
        re_road_item = re.compile(r"(Road\s*\d+):(\d+)v/(\d+)s")

        buffer = ""
        while True:
            char = self.process.stdout.read(1)
            if not char and self.process.poll() is not None:
                break
            if char in ("\r", "\n"):
                line = buffer.strip()
                buffer = ""
                if not line:
                    continue

                with self.lock:
                    if self.current_job["status"] == "cancelled":
                        break

                    self.current_job["log_lines"].append(line)

                    m_f = re_frames.search(line)
                    if m_f:
                        self.current_job["total_frames"] = int(m_f.group(1))

                    m_p = re_progress.search(line)
                    if m_p:
                        cur_f = int(m_p.group(1))
                        tot_f = int(m_p.group(2))
                        pct = int(m_p.group(3))
                        active_v = int(m_p.group(4))
                        total_c = int(m_p.group(5))
                        rest = m_p.group(6)

                        self.current_job["current_frame"] = cur_f
                        self.current_job["total_frames"] = tot_f
                        self.current_job["progress"] = min(100, max(0, pct))
                        self.current_job["active_vehicles"] = active_v
                        self.current_job["total_counted"] = total_c

                        parts = [p.strip() for p in rest.split("|")]
                        for p in parts:
                            if p.startswith("Green:"):
                                self.current_job["green_text"] = p.replace("Green:", "").strip()
                            elif p.startswith("Roads:"):
                                self.current_job["green_text"] = p
                                matches = re_road_item.findall(p)
                                for rname, veh, gsec in matches:
                                    self.current_job["live_roads"][rname] = {
                                        "vehicles": int(veh),
                                        "green_seconds": int(gsec)
                                    }
                            elif p.startswith("Forecast:"):
                                self.current_job["forecast_text"] = p.replace("Forecast:", "").strip()
                            elif p.startswith("S:") or "L:" in p:
                                self.current_job["turning_text"] = p
            else:
                buffer += char

        return_code = self.process.poll()

        with self.lock:
            if self.current_job["status"] == "cancelled":
                return

            if return_code == 0:
                summary_path = os.path.join(OUTPUTS_DIR, f"summary_{video_stem}.json")
                if not os.path.exists(summary_path):
                    summary_path = os.path.join(OUTPUTS_DIR, "summary.json")

                loaded_summary = None
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            loaded_summary = json.load(f)
                    except Exception as e:
                        print(f"[JobManager] Could not read summary JSON: {e}")

                # Transcode processed video to web-compatible H.264
                raw_processed_video = os.path.join(OUTPUTS_DIR, f"processed_{video_stem}.mp4")
                web_processed_video = os.path.join(OUTPUTS_DIR, f"web_processed_{video_stem}.mp4")
                if os.path.exists(raw_processed_video):
                    transcode_to_web_compatible(raw_processed_video, web_processed_video)

                # Invalidate in-memory cache for this video stem
                if video_stem in RESULTS_CACHE:
                    del RESULTS_CACHE[video_stem]

                self.current_job["status"] = "completed"
                self.current_job["progress"] = 100
                self.current_job["summary"] = loaded_summary
                self.current_job["output_video"] = f"processed_{video_stem}.mp4"
                self.current_job["log_lines"].append("Analysis completed successfully.")
            else:
                self.current_job["status"] = "failed"
                self.current_job["error"] = f"Traffic analysis exited with error code {return_code}."
                self.current_job["log_lines"].append(f"Analysis failed with exit code {return_code}.")


job_manager = JobManager()


# ---------------------------------------------------------------------------
# Results & Outputs Helper Functions
# ---------------------------------------------------------------------------

def get_latest_summary():
    """Finds the most recently modified summary JSON file in outputs/."""
    candidates = []
    if os.path.exists(OUTPUTS_DIR):
        for fname in os.listdir(OUTPUTS_DIR):
            if fname.startswith("summary") and fname.endswith(".json"):
                fpath = os.path.join(OUTPUTS_DIR, fname)
                candidates.append((os.path.getmtime(fpath), fpath))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    latest_path = candidates[0][1]
    stem = os.path.splitext(os.path.basename(latest_path))[0].replace("summary_", "")
    if stem == "summary":
        stem = ""
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f), stem
    except Exception:
        return None, None


def extract_forecast_chart_data(video_stem=""):
    """
    Reads the actual forecast CSV to build the Chart.js dataset with real numbers.
    Does NOT invent fake or dummy values.
    Strictly isolated: does NOT fall back to other videos' forecast CSV files.
    """
    result = {
        "has_data": False,
        "has_comparison": False,
        "reason": "No completed forecast observation intervals available yet.",
        "labels": [],
        "actual": [],
        "predicted": [],
        "arima_orders": [],
        "road_breakdown": {},
        "source_file": None
    }

    if not video_stem:
        cand = os.path.join(OUTPUTS_DIR, "forecast.csv")
        forecast_file = cand if os.path.exists(cand) else None
    else:
        cand = os.path.join(OUTPUTS_DIR, f"forecast_{video_stem}.csv")
        forecast_file = cand if os.path.exists(cand) else None

    if not forecast_file or not os.path.exists(forecast_file):
        result["reason"] = f"No forecast data file found for video '{video_stem}'." if video_stem else "No forecast data available."
        return result

    result["source_file"] = os.path.basename(forecast_file)

    if os.path.getsize(forecast_file) < 60:
        result["reason"] = (
            "Forecast file exists but contains no completed observation intervals. "
            "Auto-ARIMA requires at least 2 completed observation intervals (10 seconds of video time) to fit."
        )
        return result

    try:
        df = pd.read_csv(forecast_file)
        if df.empty or "interval_id" not in df.columns:
            result["reason"] = (
                "Forecast file exists but contains no completed observation intervals. "
                "Auto-ARIMA requires at least 2 completed observation intervals (10 seconds of video time) to fit."
            )
            return result

        # Check for road column
        has_roads = "road" in df.columns and len(df["road"].unique()) > 1

        # Extract per-road series if available
        if has_roads:
            for rname in df["road"].unique():
                rdf = df[df["road"] == rname].sort_values("interval_id")
                r_actual = []
                r_pred = []
                for _, rrow in rdf.iterrows():
                    vc = rrow.get("vehicle_count")
                    r_actual.append(int(vc) if pd.notnull(vc) else None)
                    fn = rrow.get("forecast_next")
                    try:
                        if pd.notnull(fn) and str(fn).strip() not in ("N/A", "None", ""):
                            r_pred.append(int(round(float(fn))))
                        else:
                            r_pred.append(None)
                    except (ValueError, TypeError):
                        r_pred.append(None)

                result["road_breakdown"][rname] = {
                    "actual": r_actual,
                    "predicted": r_pred
                }

            # Aggregate total traffic flow across all roads per interval
            df_agg = df.groupby("interval_id").agg({
                "timestamp_sec": "first",
                "vehicle_count": "sum",
                "forecast_next": lambda x: pd.to_numeric(x, errors="coerce").sum(min_count=1),
                "arima_order": "first"
            }).reset_index().sort_values("interval_id")
            df = df_agg

        labels = []
        actual = []
        predicted = []
        arima_orders = []
        valid_predictions = 0

        for _, row in df.iterrows():
            interval_id = int(row.get("interval_id"))
            timestamp_sec = row.get("timestamp_sec", interval_id * 5.0)
            labels.append(f"Interval {interval_id} ({timestamp_sec:.1f}s)")

            v_count = row.get("vehicle_count")
            actual.append(int(v_count) if pd.notnull(v_count) else None)

            f_val = row.get("forecast_next")
            try:
                if pd.notnull(f_val) and str(f_val).strip() not in ("N/A", "None", ""):
                    f_float = float(f_val)
                    predicted.append(int(round(f_float)))
                    valid_predictions += 1
                else:
                    predicted.append(None)
            except (ValueError, TypeError):
                predicted.append(None)

            arima_orders.append(str(row.get("arima_order", "N/A")))

        result["has_data"] = len(actual) > 0
        result["labels"] = labels
        result["actual"] = actual
        result["predicted"] = predicted
        result["arima_orders"] = arima_orders

        if valid_predictions > 0 and len(actual) > 0:
            result["has_comparison"] = True
            result["reason"] = "Actual vs. Predicted traffic comparison ready."
        elif len(actual) > 0:
            result["has_comparison"] = False
            result["reason"] = (
                f"Historical vehicle counts recorded for {len(actual)} interval(s), but Auto-ARIMA requires "
                "at least 2 observation intervals (10 seconds of video) before producing predictions."
            )
        else:
            result["has_comparison"] = False
            result["reason"] = "Insufficient video intervals collected for Auto-ARIMA forecasting."

        return result

    except Exception as e:
        result["reason"] = f"Error reading forecast file: {str(e)}"
        return result


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Renders the main dashboard HTML."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Returns current analysis status, live stats, and progress."""
    return jsonify(job_manager.get_state())


@app.route("/api/sample-videos")
def api_sample_videos():
    """Lists all sample and uploaded videos available for analysis."""
    videos = []

    if os.path.exists(SAMPLE_VIDEOS_DIR):
        for fname in sorted(os.listdir(SAMPLE_VIDEOS_DIR)):
            if is_allowed_file(fname):
                fpath = os.path.join(SAMPLE_VIDEOS_DIR, fname)
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                videos.append({
                    "name": fname,
                    "path": fpath,
                    "folder": "sample_videos",
                    "size_mb": size_mb,
                    "type": "Sample Video",
                    "media_url": f"/media/sample_videos/{fname}"
                })

    if os.path.exists(UPLOADS_DIR):
        for fname in sorted(os.listdir(UPLOADS_DIR)):
            if is_allowed_file(fname):
                fpath = os.path.join(UPLOADS_DIR, fname)
                if not any(v["name"] == fname for v in videos):
                    size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                    videos.append({
                        "name": fname,
                        "path": fpath,
                        "folder": "uploads",
                        "size_mb": size_mb,
                        "type": "Uploaded Video",
                        "media_url": f"/media/uploads/{fname}"
                    })

    return jsonify({"videos": videos})


@app.route("/api/video-info")
def api_video_info():
    """Returns metadata and playability verification for a specified video."""
    video_name = request.args.get("name", "").strip()
    folder = request.args.get("folder", "").strip()

    target_path = None
    if folder and folder in ("sample_videos", "uploads", "outputs"):
        folder_map = {
            "sample_videos": SAMPLE_VIDEOS_DIR,
            "uploads": UPLOADS_DIR,
            "outputs": OUTPUTS_DIR
        }
        cand = os.path.join(folder_map[folder], secure_filename(video_name))
        if os.path.exists(cand):
            target_path = cand

    if not target_path and video_name:
        for fdir, fname_folder in [
            (UPLOADS_DIR, "uploads"),
            (SAMPLE_VIDEOS_DIR, "sample_videos"),
            (OUTPUTS_DIR, "outputs")
        ]:
            cand = os.path.join(fdir, secure_filename(video_name))
            if os.path.exists(cand):
                target_path = cand
                folder = fname_folder
                break

    if not target_path or not os.path.exists(target_path):
        return jsonify({"exists": False, "error": f"Video file not found: {video_name}"}), 404

    size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 2)
    video_stem = os.path.splitext(os.path.basename(target_path))[0]
    has_results = os.path.exists(os.path.join(OUTPUTS_DIR, f"summary_{video_stem}.json"))

    return jsonify({
        "exists": True,
        "name": os.path.basename(target_path),
        "folder": folder,
        "size_mb": size_mb,
        "media_url": f"/media/{folder}/{os.path.basename(target_path)}",
        "has_results": has_results,
        "video_stem": video_stem
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Handles video file upload with validation."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided in request."}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"success": False, "error": "Invalid filename."}), 400

    if not is_allowed_file(filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported video format '{os.path.splitext(filename)[1]}'. Supported formats: .mp4, .avi, .mov, .mkv"
        }), 400

    dest_path = os.path.join(UPLOADS_DIR, filename)
    file.save(dest_path)
    size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)

    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": dest_path,
        "size_mb": size_mb,
        "media_url": f"/media/uploads/{filename}"
    })


@app.route("/api/start-analysis", methods=["POST"])
def api_start_analysis():
    """Starts traffic video analysis with selected options."""
    data = request.get_json(silent=True) or request.form

    video_name = data.get("video_name") or data.get("filename")
    video_path = data.get("video_path")
    mode = data.get("mode", "intersection").lower()
    roads = data.get("roads", 4)
    max_frames = data.get("max_frames")

    try:
        roads = int(roads)
    except (ValueError, TypeError):
        roads = 4

    if mode not in ("single", "intersection"):
        mode = "intersection"

    if max_frames:
        try:
            max_frames = int(max_frames)
            if max_frames <= 0:
                max_frames = None
        except (ValueError, TypeError):
            max_frames = None

    target_path = None
    if video_path and os.path.exists(video_path):
        target_path = video_path
    elif video_name:
        cand_upload = os.path.join(UPLOADS_DIR, secure_filename(video_name))
        cand_sample = os.path.join(SAMPLE_VIDEOS_DIR, secure_filename(video_name))
        if os.path.exists(cand_upload):
            target_path = cand_upload
        elif os.path.exists(cand_sample):
            target_path = cand_sample

    if not target_path or not os.path.exists(target_path):
        return jsonify({"success": False, "error": f"Video file not found: {video_name or video_path}"}), 404

    success, result = job_manager.start_job(
        video_path=target_path,
        mode=mode,
        roads=roads,
        max_frames=max_frames
    )

    if not success:
        return jsonify({"success": False, "error": result}), 409

    return jsonify({
        "success": True,
        "job_id": result,
        "message": f"Analysis started for {os.path.basename(target_path)}",
        "mode": mode,
        "roads": roads if mode == "intersection" else 1
    })


@app.route("/api/cancel-analysis", methods=["POST"])
def api_cancel_analysis():
    """Cancels the currently active analysis job if running."""
    success, msg = job_manager.cancel()
    if not success:
        return jsonify({"success": False, "error": msg}), 400
    return jsonify({"success": True, "message": msg})


RESULTS_CACHE = {}  # stem -> {"mtime": float, "payload": dict}


def get_cached_or_build_results(video_stem):
    """
    Returns cached analysis results if summary file has not changed,
    preventing repeated disk reads and pandas re-parsing.
    """
    if not video_stem:
        return None

    cand_summary = os.path.join(OUTPUTS_DIR, f"summary_{video_stem}.json")
    if not os.path.exists(cand_summary):
        return None

    mtime = os.path.getmtime(cand_summary)
    if video_stem in RESULTS_CACHE and RESULTS_CACHE[video_stem]["mtime"] == mtime:
        return RESULTS_CACHE[video_stem]["payload"]

    try:
        with open(cand_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"[Results] Error reading {cand_summary}: {e}")
        return None

    chart_data = extract_forecast_chart_data(video_stem)

    cand_video = f"processed_{video_stem}.mp4"
    cand_web_video = f"web_processed_{video_stem}.mp4"
    cand_counts = f"traffic_counts_{video_stem}.csv"
    cand_forecast = f"forecast_{video_stem}.csv"
    cand_summary_file = f"summary_{video_stem}.json"

    available_files = {}
    if os.path.exists(os.path.join(OUTPUTS_DIR, cand_web_video)):
        available_files["video"] = cand_web_video
    elif os.path.exists(os.path.join(OUTPUTS_DIR, cand_video)):
        available_files["video"] = cand_video

    for key, fname in [
        ("counts_csv", cand_counts),
        ("forecast_csv", cand_forecast),
        ("summary_json", cand_summary_file)
    ]:
        if os.path.exists(os.path.join(OUTPUTS_DIR, fname)):
            available_files[key] = fname

    payload = {
        "success": True,
        "video_stem": video_stem,
        "summary": summary,
        "forecast_chart": chart_data,
        "available_files": available_files
    }
    RESULTS_CACHE[video_stem] = {"mtime": mtime, "payload": payload}
    return payload


@app.route("/api/results")
def api_results():
    """
    Returns actual analysis results.
    Optional ?video=<stem> selects results for a specific video;
    Strictly isolated: if video is specified, it returns results ONLY for that video.
    If no results exist for that video, returns 404 without mixing data from other videos.
    """
    video_stem = request.args.get("video", "").strip()

    if video_stem:
        payload = get_cached_or_build_results(video_stem)
        if payload:
            return jsonify(payload)

        # Check in-memory job if it just completed for this video
        current_state = job_manager.get_state()
        if current_state.get("video_stem") == video_stem and current_state.get("summary"):
            chart_data = extract_forecast_chart_data(video_stem)
            available_files = {}
            if current_state.get("output_video"):
                available_files["video"] = current_state["output_video"]
            return jsonify({
                "success": True,
                "video_stem": video_stem,
                "summary": current_state["summary"],
                "forecast_chart": chart_data,
                "available_files": available_files
            })

        return jsonify({
            "success": False,
            "video_stem": video_stem,
            "message": f"No analysis results available for video '{video_stem}'. Please run an analysis."
        }), 404

    # If no video_stem requested: return latest summary if available
    summary, stem_used = get_latest_summary()
    if stem_used:
        payload = get_cached_or_build_results(stem_used)
        if payload:
            return jsonify(payload)

    return jsonify({
        "success": False,
        "message": "No analysis results available yet. Please select and analyze a video."
    }), 404


@app.route("/api/outputs")
def api_outputs():
    """Lists all files currently in the outputs/ directory grouped cleanly."""
    if not os.path.exists(OUTPUTS_DIR):
        return jsonify({"files": [], "grouped": []})

    all_files = []
    groups = {}

    for fname in sorted(os.listdir(OUTPUTS_DIR)):
        # Skip internal web_transcoded files from downloads list
        if fname.startswith("web_processed_"):
            continue

        fpath = os.path.join(OUTPUTS_DIR, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            size_bytes = stat.st_size
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{round(size_bytes / 1024, 1)} KB"
            else:
                size_str = f"{round(size_bytes / (1024 * 1024), 2)} MB"

            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

            category = "Other"
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".mp4":
                category = "Video"
            elif ext == ".csv":
                category = "Counts CSV" if "counts" in fname else "Forecast CSV"
            elif ext == ".json":
                category = "Summary JSON" if "summary" in fname else "Zones JSON"
            elif ext in (".jpg", ".png"):
                category = "Image"

            item = {
                "name": fname,
                "category": category,
                "size_bytes": size_bytes,
                "size_str": size_str,
                "modified": mtime,
                "download_url": f"/download/output/{fname}",
                "media_url": f"/media/outputs/{fname}" if ext == ".mp4" else None
            }
            all_files.append(item)

            group_key = "General"
            for prefix in ("processed_", "traffic_counts_", "forecast_", "summary_"):
                if fname.startswith(prefix):
                    stem = fname[len(prefix):]
                    stem = os.path.splitext(stem)[0]
                    group_key = stem
                    break

            if group_key not in groups:
                groups[group_key] = {
                    "stem": group_key,
                    "video": None,
                    "counts_csv": None,
                    "forecast_csv": None,
                    "summary_json": None,
                    "files": []
                }
            groups[group_key]["files"].append(item)
            if ext == ".mp4":
                groups[group_key]["video"] = item
            elif "traffic_counts" in fname:
                groups[group_key]["counts_csv"] = item
            elif "forecast" in fname:
                groups[group_key]["forecast_csv"] = item
            elif "summary" in fname:
                groups[group_key]["summary_json"] = item

    return jsonify({
        "files": all_files,
        "grouped": list(groups.values())
    })


# ---------------------------------------------------------------------------
# Media Streaming & Download Routes
# ---------------------------------------------------------------------------

@app.route("/download/<folder>/<path:filename>")
def download_file(folder, filename):
    """Safely serves output or uploaded files as downloadable attachments."""
    folder_map = {
        "output": OUTPUTS_DIR,
        "outputs": OUTPUTS_DIR,
        "upload": UPLOADS_DIR,
        "uploads": UPLOADS_DIR,
        "sample": SAMPLE_VIDEOS_DIR,
        "sample_videos": SAMPLE_VIDEOS_DIR
    }
    target_dir = folder_map.get(folder.lower())
    if not target_dir:
        abort(404)

    clean_name = os.path.basename(unquote(filename))
    file_path = os.path.join(target_dir, clean_name)

    if not os.path.exists(file_path):
        abort(404)

    return send_from_directory(target_dir, clean_name, as_attachment=True)


@app.route("/media/<folder>/<path:filename>")
def stream_media(folder, filename):
    """
    Streams video files using Flask/Werkzeug's built-in conditional range handling.
    Automatically handles HTTP 206 Partial Content, Accept-Ranges, Content-Range,
    and seeking for HTML5 video elements.
    """
    folder_map = {
        "outputs": OUTPUTS_DIR,
        "output": OUTPUTS_DIR,
        "uploads": UPLOADS_DIR,
        "upload": UPLOADS_DIR,
        "sample_videos": SAMPLE_VIDEOS_DIR,
        "sample": SAMPLE_VIDEOS_DIR
    }
    target_dir = folder_map.get(folder.lower())
    if not target_dir:
        abort(404)

    clean_name = os.path.basename(unquote(filename))

    # If asking for processed_<stem>.mp4 and web_processed_<stem>.mp4 exists, serve the web-compatible version
    if folder.lower() in ("output", "outputs") and clean_name.startswith("processed_"):
        web_name = f"web_{clean_name}"
        if os.path.exists(os.path.join(target_dir, web_name)):
            clean_name = web_name

    file_path = os.path.join(target_dir, clean_name)
    if not os.path.exists(file_path):
        abort(404)

    # Determine MIME type
    ext = os.path.splitext(clean_name)[1].lower()
    mimes = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska"
    }
    mimetype = mimes.get(ext, "video/mp4")

    # Werkzeug send_from_directory natively handles Range headers when conditional=True
    return send_from_directory(
        target_dir,
        clean_name,
        mimetype=mimetype,
        conditional=True
    )


# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  Traffic Video Analyzer - Website Dashboard")
    print(f"  Starting server on http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
