"""
Unit tests for the CrowdDetector module.
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import CrowdDetector


@pytest.fixture
def detector():
    """Create a CrowdDetector instance."""
    return CrowdDetector(max_capacity=200)


@pytest.fixture
def blank_image():
    """Create a blank test image (no people expected)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestCrowdDetector:
    """Tests for CrowdDetector."""

    def test_detect_people_returns_dict(self, detector, blank_image):
        """Detection should return a properly structured dictionary."""
        result = detector.detect_people(blank_image)
        assert isinstance(result, dict)
        assert 'count' in result
        assert 'boxes' in result
        assert 'confidences' in result
        assert isinstance(result['count'], int)
        assert result['count'] >= 0

    def test_compute_occupancy(self, detector):
        """Occupancy computation should return correct levels."""
        # Low density
        occ = detector.compute_occupancy(20)
        assert occ['density_level'] == "Low"
        assert occ['occupancy_ratio'] == 0.18

        # High density
        occ = detector.compute_occupancy(60)
        assert occ['density_level'] == "High"

        # Overcrowded
        occ = detector.compute_occupancy(200)
        assert occ['density_level'] == "Overcrowded"

    def test_annotate_image(self, detector, blank_image):
        """Annotation should return a valid image array."""
        detection_result = {
            'boxes': [[100, 100, 200, 300]],
            'confidences': [0.95]
        }
        annotated = detector.annotate_image(blank_image, detection_result)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == blank_image.shape