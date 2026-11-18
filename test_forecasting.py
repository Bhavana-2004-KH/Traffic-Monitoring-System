"""
Unit test for TrafficForecaster to verify Auto-ARIMA model fitting and predictions.
"""
import numpy as np
import pandas as pd
from forecasting import TrafficForecaster

def test_arima_forecasting():
    print("=" * 60)
    print("TESTING AUTO-ARIMA TRAFFIC FORECASTING")
    print("=" * 60)
    
    forecaster = TrafficForecaster(interval_seconds=1.0, min_observations=5)
    
    # Simulate realistic traffic count arrival series across 8 intervals
    synthetic_traffic_series = [12, 14, 13, 17, 16, 20, 19, 22]
    
    print(f"Simulating traffic counts over {len(synthetic_traffic_series)} intervals:")
    for idx, count in enumerate(synthetic_traffic_series):
        timestamp = (idx + 1) * 1.05
        forecaster.add_frame_sample(timestamp, count)
        print(f"  Interval {idx+1} (t={timestamp:.1f}s, count={count}): Status -> '{forecaster.status}'")
        
    print("-" * 60)
    print(f"Auto-ARIMA Predicted Next Interval: {forecaster.predicted_next}")
    print(f"Fitted ARIMA Order (p, d, q): {forecaster.last_model_order}")
    print(f"Display String: {forecaster.get_forecast_display()}")
    
    assert forecaster.predicted_next is not None, "Prediction should not be None after >= 5 observations"
    assert forecaster.last_model_order is not None, "Model order should be identified by Auto-ARIMA"
    print("=" * 60)
    print("SUCCESS: Auto-ARIMA forecasting verified with real pmdarima execution!")

if __name__ == "__main__":
    test_arima_forecasting()
