from __future__ import annotations

"""List serial ports visible to Python.

Useful on macOS after pairing BWT901CL Bluetooth sensors. The ports usually look
like `/dev/cu.*` or `/dev/tty.*`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> None:
    """Print available serial ports."""

    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required. Install it with `pip install pyserial`.") from exc

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        print(port.device)
        print(f"  description: {port.description}")
        print(f"  hwid: {port.hwid}")


if __name__ == "__main__":
    main()
