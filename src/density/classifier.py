"""
Turns a raw person count into a passenger-friendly density label.
Kept separate from detector.py so you can change the rules (or later
swap in a trained classifier) without touching detection code.
"""
from dataclasses import dataclass

from src.config import COACH_CAPACITY, DENSITY_THRESHOLDS


@dataclass
class DensityResult:
    count: int
    occupancy_ratio: float  # 0.0 - 1.0+
    level: str


def classify_density(count: int, capacity: int = COACH_CAPACITY) -> DensityResult:
    ratio = count / capacity if capacity > 0 else 0.0

    level = "Overcrowded"
    for label, upper_bound in DENSITY_THRESHOLDS.items():
        if ratio <= upper_bound:
            level = label
            break

    return DensityResult(count=count, occupancy_ratio=round(ratio, 2), level=level)


if __name__ == "__main__":
    for test_count in [20, 90, 160, 210]:
        r = classify_density(test_count)
        print(f"count={r.count:>4}  ratio={r.occupancy_ratio:>4}  level={r.level}")
