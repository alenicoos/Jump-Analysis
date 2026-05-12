from __future__ import annotations

"""CLI to convert the raw mocap dataset into the project reference CSV."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_analysis.data.mocap_dataset import KinematicDataConverter


def main() -> None:
    """Converte tutti i soggetti o un singolo soggetto di debug."""

    parser = argparse.ArgumentParser(
        description="Convert the 183-athlete Kinematic_Data dataset to the 37 front-view LESS features."
    )
    parser.add_argument("--root", default="/Users/ale/Kinematic_Data", help="Path to Kinematic_Data.")
    parser.add_argument("--output", default="mocap_front_37_features.csv", help="Output CSV.")
    parser.add_argument("--subject", help="Convert only one subject id for debugging.")
    args = parser.parse_args()

    converter = KinematicDataConverter.from_path(args.root)
    if args.subject:
        # Modalita' rapida per controllare un soggetto senza convertire 26GB.
        row = converter.convert_subject(args.subject)
        import pandas as pd

        frame = pd.DataFrame([row])
    else:
        frame = converter.convert_all()
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
