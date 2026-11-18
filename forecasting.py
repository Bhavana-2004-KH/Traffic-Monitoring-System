"""
Forecasting module using pmdarima.auto_arima.
Collects time-series traffic observations over discrete intervals,
fits an Auto-ARIMA model, and predicts upcoming vehicle demand.
"""
import os
import pandas as pd
import numpy as np
import pmdarima as pm
import config

class TrafficForecaster:
    def __init__(self, interval_seconds=None, min_observations=None, name="Overall"):
        self.interval_seconds = interval_seconds if interval_seconds is not None else config.FORECAST_INTERVAL_SECONDS
        self.min_observations = min_observations if min_observations is not None else config.MIN_FORECAST_OBSERVATIONS
        self.name = name
        
        # Historical interval records: list of dicts {"interval_id", "timestamp_sec", "vehicle_count", "forecast_next", "arima_order"}
        self.observations = []
        
        # Current interval state
        self.current_interval_id = 0
        self.interval_samples = []
        self.interval_track_ids = set()
        self.last_interval_timestamp = 0.0
        
        self.predicted_next = None
        self.status = "Preparing forecast..."
        self.last_model_order = None
        
    def add_frame_sample(self, timestamp_seconds, vehicle_count, active_track_ids=None):
        """
        Accumulates vehicle count samples and track IDs within the current observation interval.
        Triggers Auto-ARIMA fitting once an interval completes and sufficient history exists.
        """
        self.interval_samples.append(vehicle_count)
        if active_track_ids:
            self.interval_track_ids.update(active_track_ids)
            
        # Check if interval elapsed
        if timestamp_seconds - self.last_interval_timestamp >= self.interval_seconds:
            self.current_interval_id += 1
            
            # Determine actual vehicle count for this completed interval:
            # If unique track IDs were observed during the interval, use that count;
            # otherwise use the rounded mean of instantaneous counts.
            if self.interval_track_ids:
                actual_count = len(self.interval_track_ids)
            elif self.interval_samples:
                actual_count = int(round(np.mean(self.interval_samples)))
            else:
                actual_count = int(vehicle_count)
                
            self.observations.append({
                "interval_id": self.current_interval_id,
                "timestamp_sec": round(timestamp_seconds, 2),
                "vehicle_count": actual_count,
                "forecast_next": None,
                "arima_order": None
            })
            
            # Reset current interval accumulators
            self.interval_samples = []
            self.interval_track_ids = set()
            self.last_interval_timestamp = timestamp_seconds
            
            # Attempt Auto-ARIMA prediction only upon interval completion
            self._fit_and_forecast()
            
    def _fit_and_forecast(self):
        """Fits Auto-ARIMA on observed traffic series and predicts next interval count."""
        if len(self.observations) < self.min_observations:
            self.status = "Preparing forecast..."
            self.predicted_next = None
            return
            
        df = pd.DataFrame(self.observations)
        series = df["vehicle_count"].values.astype(float)
        
        try:
            # Check if all historical values are identical (constant series edge case)
            if np.all(series == series[0]):
                self.predicted_next = int(series[0])
                self.last_model_order = (0, 0, 0)
            else:
                # Fit Auto-ARIMA model
                model = pm.auto_arima(
                    series,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    d=None,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_order=5
                )
                
                forecast = model.predict(n_periods=config.FORECAST_STEPS)
                pred_float = float(forecast[0] if isinstance(forecast, (list, np.ndarray, pd.Series)) else forecast)
                
                # Traffic count cannot be negative
                self.predicted_next = max(0, int(round(pred_float)))
                self.last_model_order = model.order
                
            self.status = f"Next Traffic: {self.predicted_next} vehicles"
            if self.observations:
                self.observations[-1]["forecast_next"] = self.predicted_next
                self.observations[-1]["arima_order"] = str(self.last_model_order)
                
        except Exception as e:
            # Fallback to moving average if numerical instability occurs
            moving_avg = int(round(np.mean(series[-3:])))
            self.predicted_next = max(0, moving_avg)
            self.last_model_order = "MA(3)"
            self.status = f"Next Traffic: {self.predicted_next} vehicles"
            if self.observations:
                self.observations[-1]["forecast_next"] = self.predicted_next
                self.observations[-1]["arima_order"] = "MA(3)"
                
    def get_forecast_display(self):
        """Returns the formatted status text for video HUD and dashboard."""
        return self.status

    def get_final_status(self):
        """Returns descriptive terminal status including insufficient intervals check."""
        if self.predicted_next is not None:
            return f"Next Traffic: {self.predicted_next}"
        return f"Insufficient traffic intervals for Auto-ARIMA ({len(self.observations)}/{self.min_observations} intervals collected)"

    def export_csv(self, file_path=None, road=None):
        """Saves historical interval observations and forecasts to CSV."""
        if not file_path:
            file_path = os.path.join(config.OUTPUT_FOLDER, "forecast.csv")
            
        r_name = road or self.name
        rows = []
        for obs in self.observations:
            rows.append({
                "road": r_name,
                "interval_id": obs["interval_id"],
                "timestamp_sec": obs["timestamp_sec"],
                "vehicle_count": obs["vehicle_count"],
                "forecast_next": obs.get("forecast_next") if obs.get("forecast_next") is not None else "N/A",
                "arima_order": obs.get("arima_order") if obs.get("arima_order") is not None else "N/A"
            })
            
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)
        else:
            df = pd.DataFrame(columns=["road", "interval_id", "timestamp_sec", "vehicle_count", "forecast_next", "arima_order"])
            df.to_csv(file_path, index=False)
            
        return file_path

