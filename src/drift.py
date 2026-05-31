from collections import deque
from dataclasses import dataclass, field

import numpy as np
from river.drift import ADWIN


@dataclass
class RollingMAEDetector:
    window: int = 21
    threshold: float = 0.42
    errors: deque = field(default_factory=deque)

    def update(self, error: float) -> bool:
        self.errors.append(abs(error))

        if len(self.errors) > self.window:
            self.errors.popleft()

        if len(self.errors) < self.window:
            return False

        return np.mean(self.errors) > self.threshold

    def reset(self):
        self.errors.clear()


@dataclass
class PageHinkleyDetector:
    delta: float = 0.005
    threshold: float = 0.5
    alpha: float = 0.999

    mean: float = 0.0
    cumulative_sum: float = 0.0
    min_cumulative_sum: float = 0.0
    n: int = 0

    def update(self, error: float) -> bool:
        x = abs(error)
        self.n += 1

        if self.n == 1:
            self.mean = x
            return False

        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.cumulative_sum += x - self.mean - self.delta
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)

        return self.cumulative_sum - self.min_cumulative_sum > self.threshold

    def reset(self):
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.n = 0


@dataclass
class ADWINDetector:
    delta: float = 0.002

    def __post_init__(self):
        if ADWIN is None:
            self.detector = None
        else:
            self.detector = ADWIN(delta=self.delta)

    def update(self, error: float) -> bool:
        if self.detector is None:
            return False

        self.detector.update(abs(error))
        return self.detector.drift_detected

    def reset(self):
        if ADWIN is not None:
            self.detector = ADWIN(delta=self.delta)


@dataclass
class DriftMonitor:
    mae_detector: RollingMAEDetector = field(default_factory=RollingMAEDetector)
    ph_detector: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)
    adwin_detector: ADWINDetector = field(default_factory=ADWINDetector)
    min_votes: int = 2

    def update(self, date, y_true, y_pred, is_holiday: bool = False) -> dict:
        if is_holiday:
            return {
                "date": date,
                "error": None,
                "rolling_mae_alarm": False,
                "page_hinkley_alarm": False,
                "adwin_alarm": False,
                "drift_alarm": False,
                "reason": "holiday_skipped",
            }

        error = y_true - y_pred

        rolling_alarm = self.mae_detector.update(error)
        ph_alarm = self.ph_detector.update(error)
        adwin_alarm = self.adwin_detector.update(error)

        votes = sum([rolling_alarm, ph_alarm, adwin_alarm])
        drift_alarm = votes >= self.min_votes

        return {
            "date": date,
            "error": error,
            "abs_error": abs(error),
            "rolling_mae_alarm": rolling_alarm,
            "page_hinkley_alarm": ph_alarm,
            "adwin_alarm": adwin_alarm,
            "votes": votes,
            "drift_alarm": drift_alarm,
        }

    def reset(self):
        self.mae_detector.reset()
        self.ph_detector.reset()
        self.adwin_detector.reset()
