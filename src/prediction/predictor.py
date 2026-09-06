"""
Predicts expected occupancy for a given train number + date, using
historical patterns (day of week, weekend/holiday, time of day).

Starts as a simple, explainable model (linear regression on a handful
of engineered features) rather than something exotic. Easy to defend
in a viva, easy to debug, and good enough for a prototype. Swap in a
gradient-boosted model (XGBoost/LightGBM) later once you have real data
and want to squeeze out more accuracy.
"""
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import HISTORY_CSV


@dataclass
class PredictionResult:
    train_number: str
    travel_date: str
    predicted_count: int
    predicted_ratio: float


class OccupancyPredictor:
    def __init__(self, history_csv=HISTORY_CSV):
        self.history_csv = history_csv
        self.model = LinearRegression()
        self._trained = False

    def _load_history(self) -> pd.DataFrame:
        df = pd.read_csv(self.history_csv, parse_dates=["date"])
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        return df

    def train(self):
        df = self._load_history()
        features = df[["day_of_week", "is_weekend", "is_holiday"]]
        target = df["passenger_count"]
        self.model.fit(features, target)
        self._trained = True

    def predict(self, train_number: str, travel_date: str, is_holiday: bool = False) -> PredictionResult:
        if not self._trained:
            self.train()

        d = datetime.strptime(travel_date, "%Y-%m-%d").date()
        day_of_week = d.weekday()
        is_weekend = 1 if day_of_week in (5, 6) else 0

        features = pd.DataFrame(
            [[day_of_week, is_weekend, int(is_holiday)]],
            columns=["day_of_week", "is_weekend", "is_holiday"],
        )
        predicted_count = max(0, int(self.model.predict(features)[0]))

        from src.config import COACH_CAPACITY
        ratio = round(predicted_count / COACH_CAPACITY, 2)

        return PredictionResult(
            train_number=train_number,
            travel_date=travel_date,
            predicted_count=predicted_count,
            predicted_ratio=ratio,
        )


if __name__ == "__main__":
    predictor = OccupancyPredictor()
    result = predictor.predict(train_number="16526", travel_date="2026-07-11")
    print(result)
