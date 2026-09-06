"""
CLI script to run crowd detection on an image.
"""

import os
import sys
import argparse
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import CrowdDetector


def main():
    parser = argparse.ArgumentParser(
        description="Detect crowd density in a train image"
    )

    parser.add_argument(
        "image_path",
        help="Path to the input image"
    )

    parser.add_argument(
        "--output", "-o",
        help="Path to save annotated image (default: adds '_annotated' suffix)"
    )

    parser.add_argument(
        "--capacity", "-c",
        type=int,
        default=200,
        help="Maximum passenger capacity (default: 200)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    print("Loading detector...")
    detector = CrowdDetector(max_capacity=args.capacity)

    print(f"Processing image: {args.image_path}")
    result = detector.process_image(args.image_path)

    detection = result["detection"]
    occupancy = result["occupancy"]

    print("\n" + "=" * 50)
    print("  DETECTION RESULTS")
    print("=" * 50)
    print(f"  People detected : {occupancy['detected_count']}")
    print(f"  Estimated people: {occupancy['count']}")
    print(f"  Max capacity    : {occupancy['max_capacity']}")
    print(f"  Occupancy ratio : {occupancy['occupancy_ratio']}")
    print(f"  Occupancy %     : {occupancy['occupancy_percent']}%")
    print(f"  Density level   : {occupancy['density_level']}")
    print("=" * 50)

    # Save annotated image
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image_path)
        output_path = f"{base}_annotated{ext}"

    cv2.imwrite(output_path, result["annotated_image"])
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    main()