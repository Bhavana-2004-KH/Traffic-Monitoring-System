"""
Signal Controller module for simulated adaptive traffic signal timing.
Translates forecasted vehicle demand into recommended green signal duration.

Green time is recalculated once per SIGNAL CYCLE (every config.SIGNAL_CYCLE_FRAMES frames)
using an Exponential Moving Average of observed demand so that small detection fluctuations
do not cause the displayed green time to jump on every frame.
"""
import config


class AdaptiveSignalController:
    """
    Simulates adaptive traffic signal green light calculation based on demand.

    Formula (applied once per cycle):
      smoothed_demand = EMA(alpha, observed_demand)
      GreenTime = clamp(BaseGreen + (smoothed_demand * ScaleFactor), MinGreen, MaxGreen)

    Where:
      - BaseGreen     = config.BASE_GREEN_LIGHT  (default 25 s)
      - ScaleFactor   = config.SECONDS_PER_VEHICLE (default 2.5 s/vehicle)
      - MinGreen      = config.MIN_GREEN_LIGHT   (default 20 s)
      - MaxGreen      = config.MAX_GREEN_LIGHT   (default 90 s)
      - CycleFrames   = config.SIGNAL_CYCLE_FRAMES (default 90 frames ≈ 3 s @ 30 fps)
      - Alpha (EMA)   = config.SIGNAL_SMOOTHING_ALPHA (default 0.3)
    """

    def __init__(
        self,
        min_green=config.MIN_GREEN_LIGHT,
        max_green=config.MAX_GREEN_LIGHT,
        base_green=config.MIN_GREEN_LIGHT,
        scale_factor=config.SECONDS_PER_VEHICLE,
        cycle_frames=config.SIGNAL_CYCLE_FRAMES,
        smoothing_alpha=config.SIGNAL_SMOOTHING_ALPHA,
        max_change_per_cycle=getattr(config, "MAX_GREEN_CHANGE_PER_CYCLE", 5),
        yellow_duration=getattr(config, "YELLOW_LIGHT_SECONDS", 3),
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.base_green = base_green
        self.scale_factor = scale_factor
        self.cycle_frames = cycle_frames
        self.alpha = smoothing_alpha
        self.max_change = max_change_per_cycle
        self.yellow_duration = yellow_duration

        # --- Single-road state ---
        self.current_green_time = base_green   # stable; only updated at cycle boundary
        self._ema_demand = None                # EMA-smoothed demand
        self._cycle_frame_counter = 0          # frames elapsed in current cycle
        self._cycle_demand_accumulator = []    # raw demand samples within current cycle

        # --- Intersection / per-road state ---
        # road_name -> {ema_demand, current_green_time, samples}
        self._road_states = {}
        self.active_road_index = 0
        self.current_phase = "GREEN"           # "GREEN", "YELLOW"
        self.phase_elapsed_seconds = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, predicted_vehicles=None, current_vehicles=0, frame_idx=None):
        """
        Returns the current stable green time for single-road mode.

        Recalculates only when a cycle boundary is crossed.
        Between cycles the previously computed green time is returned unchanged.

        Args:
            predicted_vehicles: Auto-ARIMA forecast (preferred); None while warming up.
            current_vehicles:   Fallback live count used when forecast is unavailable.
            frame_idx:          Current frame index; if None, cycle counting is skipped
                                (backward-compatible – caller can omit it).
        """
        demand = predicted_vehicles if predicted_vehicles is not None else current_vehicles

        # Accumulate demand sample for averaging within the cycle
        self._cycle_demand_accumulator.append(demand)
        self._cycle_frame_counter += 1

        # Recalculate only at cycle boundaries
        if self._cycle_frame_counter >= self.cycle_frames:
            cycle_avg = sum(self._cycle_demand_accumulator) / max(1, len(self._cycle_demand_accumulator))
            self._cycle_demand_accumulator = []
            self._cycle_frame_counter = 0

            # Update EMA
            if self._ema_demand is None:
                self._ema_demand = cycle_avg
            else:
                self._ema_demand = self.alpha * cycle_avg + (1 - self.alpha) * self._ema_demand

            # Compute and store stable green time with max-change clamping
            raw = self.base_green + (self._ema_demand * self.scale_factor)
            target = int(round(max(self.min_green, min(self.max_green, raw))))
            delta = target - self.current_green_time
            if delta > self.max_change:
                self.current_green_time += self.max_change
            elif delta < -self.max_change:
                self.current_green_time -= self.max_change
            else:
                self.current_green_time = target

        return {
            "green_seconds": self.current_green_time,
            "demand": demand,
            "source": "forecast" if predicted_vehicles is not None else "current",
            "display": f"GREEN LIGHT: {self.current_green_time} SECONDS",
        }

    def calculate_roads(self, road_demands, fps=30.0):
        """
        Calculates stable adaptive green signal durations and phase tracking
        for each road in INTERSECTION mode.
        Each road has its own independent EMA state so roads respond independently to demand.

        Args:
            road_demands: dict {road_name: vehicle_count}
            fps: video frames per second for phase duration tracking

        Returns:
            dict {road_name: {"vehicles": int, "green_seconds": int, "signal_state": str, "phase_remaining": int}}
        """
        # Initialise state for any new roads
        for road_name in road_demands:
            if road_name not in self._road_states:
                self._road_states[road_name] = {
                    "ema_demand": None,
                    "current_green_time": self.base_green,
                    "samples": [],
                    "frame_counter": 0,
                }

        # Accumulate per-road samples; advance each road's cycle counter
        for road_name, demand in road_demands.items():
            rs = self._road_states[road_name]
            rs["samples"].append(demand)
            rs["frame_counter"] += 1

            if rs["frame_counter"] >= self.cycle_frames:
                cycle_avg = sum(rs["samples"]) / max(1, len(rs["samples"]))
                rs["samples"] = []
                rs["frame_counter"] = 0

                if rs["ema_demand"] is None:
                    rs["ema_demand"] = cycle_avg
                else:
                    rs["ema_demand"] = self.alpha * cycle_avg + (1 - self.alpha) * rs["ema_demand"]

                raw = self.base_green + (rs["ema_demand"] * self.scale_factor)
                target = int(round(max(self.min_green, min(self.max_green, raw))))
                delta = target - rs["current_green_time"]
                if delta > self.max_change:
                    rs["current_green_time"] += self.max_change
                elif delta < -self.max_change:
                    rs["current_green_time"] -= self.max_change
                else:
                    rs["current_green_time"] = target

        # Advance intersection phase tracking
        dt = 1.0 / max(1.0, float(fps))
        self.phase_elapsed_seconds += dt

        road_names = list(road_demands.keys())
        if road_names:
            if self.active_road_index >= len(road_names):
                self.active_road_index = 0
            active_road = road_names[self.active_road_index]
            active_green = self._road_states[active_road]["current_green_time"]

            if self.current_phase == "GREEN":
                if self.phase_elapsed_seconds >= active_green:
                    self.current_phase = "YELLOW"
                    self.phase_elapsed_seconds = 0.0
            elif self.current_phase == "YELLOW":
                if self.phase_elapsed_seconds >= self.yellow_duration:
                    self.current_phase = "GREEN"
                    self.phase_elapsed_seconds = 0.0
                    self.active_road_index = (self.active_road_index + 1) % len(road_names)
                    active_road = road_names[self.active_road_index]
                    active_green = self._road_states[active_road]["current_green_time"]
        else:
            active_road = None
            active_green = self.base_green

        # Build result with green times and signal states
        res = {}
        for rname, demand in road_demands.items():
            r_green = self._road_states[rname]["current_green_time"]
            if rname == active_road:
                sig_state = self.current_phase
                if self.current_phase == "GREEN":
                    rem = max(0, int(round(active_green - self.phase_elapsed_seconds)))
                else:
                    rem = max(0, int(round(self.yellow_duration - self.phase_elapsed_seconds)))
            else:
                sig_state = "RED"
                rem = 0

            res[rname] = {
                "vehicles": demand,
                "green_seconds": r_green,
                "signal_state": sig_state,
                "phase_remaining": rem,
            }

        return res

    def get_display(self):
        """Returns standard display string for the current stable green time."""
        return f"GREEN LIGHT: {self.current_green_time} SECONDS"
