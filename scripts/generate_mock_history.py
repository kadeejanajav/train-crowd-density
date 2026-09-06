"""
Generate mock historical passenger data for testing and training.
"""

import os
import sys
import random
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_DIR, MAX_CAPACITY


def generate_mock_data(days=365, train_number="16526"):
    """
    Generate realistic mock passenger count data.

    Patterns simulated:
    - Morning rush (7-9 AM): High occupancy
    - Evening rush (5-7 PM): High occupancy
    - Midday: Moderate occupancy
    - Late night: Low occupancy
    - Weekends: Lower overall occupancy
    """
    records = []
    start_date = datetime.now() - timedelta(days=days)

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        day_of_week = current_date.weekday()  # 0=Mon, 6=Sun
        is_weekend = day_of_week >= 5

        # Generate data for operating hours (5 AM to 11 PM)
        for hour in range(5, 24):
            # Base passenger count varies by time of day
            if 7 <= hour <= 9:  # Morning rush
                base = int(MAX_CAPACITY * random.uniform(0.6, 0.95))
            elif 17 <= hour <= 19:  # Evening rush
                base = int(MAX_CAPACITY * random.uniform(0.55, 0.90))
            elif 10 <= hour <= 16:  # Midday
                base = int(MAX_CAPACITY * random.uniform(0.25, 0.50))
            elif 20 <= hour <= 23:  # Evening
                base = int(MAX_CAPACITY * random.uniform(0.10, 0.30))
            else:  # Early morning (5-6)
                base = int(MAX_CAPACITY * random.uniform(0.05, 0.20))

            # Reduce weekend traffic
            if is_weekend:
                base = int(base * random.uniform(0.4, 0.7))

            # Add random noise
            noise = random.randint(-10, 10)
            passenger_count = max(0, min(base + noise, MAX_CAPACITY))

            records.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'hour': hour,
                'day_of_week': day_of_week,
                'month': current_date.month,
                'train_number': train_number,
                'passenger_count': passenger_count,
                'is_weekend': int(is_weekend)
            })

    return pd.DataFrame(records)


def main():
    """Generate and save mock data."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Generating mock historical data...")
    df = generate_mock_data(days=365)

    output_path = os.path.join(PROCESSED_DIR, "history.csv")
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} records")
    print(f"Saved to: {output_path}")
    print(f"\nSample data:")
    print(df.head(10).to_string(index=False))
    print(f"\nPassenger count statistics:")
    print(df['passenger_count'].describe())


if __name__ == "__main__":
    main()