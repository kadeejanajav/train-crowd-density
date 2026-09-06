"""
Occupancy prediction module.
Uses historical data to predict future crowd density using simple ML.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.config import HISTORY_FILE


class OccupancyPredictor:
    """Predicts future occupancy based on historical passenger data."""

    def __init__(self, history_file=None):
        """
        Initialize the predictor.

        Args:
            history_file: Path to CSV with historical data.
        """
        self.history_file = history_file or HISTORY_FILE
        self.model = None
        self.history_df = None
        self.is_trained = False

    def load_history(self):
        """Load historical data from CSV."""
        if not os.path.exists(self.history_file):
            raise FileNotFoundError(
                f"History file not found: {self.history_file}\n"
                f"Run 'python scripts/generate_mock_history.py' first."
            )

        self.history_df = pd.read_csv(self.history_file)

        # Parse datetime features
        if 'date' in self.history_df.columns:
            self.history_df['date'] = pd.to_datetime(self.history_df['date'])
            self.history_df['day_of_week'] = self.history_df['date'].dt.dayofweek
            self.history_df['month'] = self.history_df['date'].dt.month
            self.history_df['day_of_year'] = self.history_df['date'].dt.dayofyear

        return self.history_df

    def _prepare_features(self, df):
        """Extract feature columns for training/prediction."""
        feature_cols = ['hour', 'day_of_week', 'month']
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            raise ValueError("No usable feature columns found in data.")
        return df[available]

    def train(self):
        """Train the prediction model on historical data."""
        if self.history_df is None:
            self.load_history()

        X = self._prepare_features(self.history_df)
        y = self.history_df['passenger_count']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        self.is_trained = True

        return {
            'train_r2': round(train_score, 4),
            'test_r2': round(test_score, 4),
            'n_samples': len(X),
            'features': list(X.columns)
        }

    def predict(self, hour, day_of_week=None, month=None):
        """
        Predict passenger count for given conditions.

        Args:
            hour: Hour of the day (0-23).
            day_of_week: Day of week (0=Mon, 6=Sun). Defaults to today.
            month: Month (1-12). Defaults to current month.

        Returns:
            dict with predicted count, occupancy ratio, and density level.
        """
        if not self.is_trained:
            self.train()

        now = datetime.now()
        if day_of_week is None:
            day_of_week = now.weekday()
        if month is None:
            month = now.month

        features = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month
        }])

        feature_cols = [c for c in self.model.feature_names_in_ if c in features.columns]
        prediction = self.model.predict(features[feature_cols])[0]
        prediction = max(0, int(round(prediction)))

        from src.config import MAX_CAPACITY, DENSITY_THRESHOLDS
        ratio = min(prediction / MAX_CAPACITY, 1.0)
        level = "Unknown"
        for label, (low, high) in DENSITY_THRESHOLDS.items():
            if low <= ratio < high:
                level = label
                break
        if ratio >= 1.0:
            level = "Overcrowded"

        return {
            'predicted_count': prediction,
            'occupancy_ratio': round(ratio, 4),
            'occupancy_percent': round(ratio * 100, 2),
            'density_level': level,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month
        }

    def predict_day(self, day_of_week=None, month=None):
        """Predict hourly occupancy for an entire day."""
        predictions = []
        for hour in range(5, 24):  # 5 AM to 11 PM
            pred = self.predict(hour, day_of_week, month)
            predictions.append(pred)
        return predictions