from __future__ import annotations

"""Convert mocap drop-jump shank orientations to a CSV time series."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.data.mocap_knee_orientation import MocapKneeOrientationExporter


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Export mocap knee/shank pitch-roll-yaw targets for drop jumps.")
    parser.add_argument("--root", default="/Users/ale/Kinematic_Data", help="Path to Kinematic_Data.")
    parser.add_argument("--output", default="mocap_knee_orientation_timeseries.csv", help="Output CSV path.")
    args = parser.parse_args()

    exporter = MocapKneeOrientationExporter.from_path(args.root)
    frame = exporter.export_all()
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} frame rows to {args.output}")


if __name__ == "__main__":
    main()
