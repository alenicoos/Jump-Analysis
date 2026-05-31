from __future__ import annotations

"""Live reader for WITMOTION BWT901CL sensors.

The BWT901CL exposes WIT serial packets over Bluetooth/serial. The angle packet
has this structure:

`55 53 roll_l roll_h pitch_l pitch_h yaw_l yaw_h version_l version_h checksum`

Roll, pitch and yaw are signed int16 values scaled as `value / 32768 * 180`.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .imu_timeseries import ImuOrientationSeries

ANGLE_PACKET_ID = 0x53
DEFAULT_BAUD_RATE = 115200
UNLOCK_COMMAND = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
SAVE_COMMAND = bytes([0xFF, 0xAA, 0x00, 0x00, 0x00])

# WIT register 0x02 controls returned data. Bit 0x08 enables angle output.
ANGLE_OUTPUT_COMMAND = bytes([0xFF, 0xAA, 0x02, 0x08, 0x00])

# WIT register 0x03 controls return rate. Value 0x06 is commonly 10 Hz.
RATE_10HZ_COMMAND = bytes([0xFF, 0xAA, 0x03, 0x06, 0x00])


@dataclass
class Bwt901clSample:
    """One decoded BWT901CL angle sample."""

    timestamp_s: float
    pitch_deg: float | None
    roll_deg: float | None
    yaw_deg: float | None
    accel_x: float | None = None
    accel_y: float | None = None
    accel_z: float | None = None
    gyro_x: float | None = None
    gyro_y: float | None = None
    gyro_z: float | None = None


class Bwt901clReader:
    """Threaded reader for one BWT901CL serial/Bluetooth port."""

    def __init__(
        self,
        port: str,
        baud_rate: int = DEFAULT_BAUD_RATE,
        timeout_s: float = 0.05,
        name: str = "bwt901cl",
        configure_angle_output: bool = False,
        backend: Literal["witmotion", "manual"] = "witmotion",
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self.name = name
        self.configure_angle_output = configure_angle_output
        self.backend = backend
        self.samples: list[Bwt901clSample] = []
        self.raw_byte_count = 0
        self.packet_count = 0
        self.angle_packet_count = 0
        self.checksum_error_count = 0
        self.packet_type_counts: dict[int, int] = {}
        self.raw_preview = bytearray()
        self._serial = None
        self._imu = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Open the serial port and start collecting samples."""

        if self._thread is not None and self._thread.is_alive():
            return
        if self.backend == "witmotion":
            try:
                from witmotion import IMU
                from witmotion import protocol
            except ImportError as exc:
                raise RuntimeError("witmotion is required. Install it with `pip install witmotion`.") from exc

            self._imu = IMU(self.port, baudrate=self.baud_rate)
            if self.configure_angle_output:
                self._imu.set_messages_enabled(
                    {
                        protocol.AccelerationMessage,
                        protocol.AngularVelocityMessage,
                        protocol.AngleMessage,
                    }
                )
                self._imu.set_update_rate(10)
                self._imu.save_configuration()
        else:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is required. Install it with `pip install pyserial`.") from exc

            self._serial = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=self.timeout_s,
                write_timeout=1.0,
                rtscts=False,
                dsrdtr=False,
            )
            try:
                self._serial.dtr = False
                self._serial.rts = False
            except Exception:
                pass
            if self.configure_angle_output:
                configure_bwt901cl_angle_output(self._serial)
        self._stop_event.clear()
        target = self._witmotion_read_loop if self.backend == "witmotion" else self._manual_read_loop
        self._thread = threading.Thread(target=target, name=f"{self.name}-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop collection and close the serial port."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._imu is not None:
            # The third-party `witmotion.IMU.close()` stops only its receive
            # thread. On macOS Bluetooth SPP we also need to close the
            # underlying pyserial handle explicitly, otherwise the next run can
            # reconnect but receive no stream from the HC-06 port.
            imu_serial = getattr(self._imu, "ser", None)
            self._imu.close()
            if imu_serial is not None and getattr(imu_serial, "is_open", False):
                try:
                    imu_serial.reset_input_buffer()
                    imu_serial.reset_output_buffer()
                    imu_serial.dtr = False
                    imu_serial.rts = False
                except Exception:
                    pass
                imu_serial.close()
            self._imu = None
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self._serial = None
        self._thread = None

    def clear(self) -> None:
        """Remove collected samples."""

        with self._lock:
            self.samples.clear()
            self.raw_byte_count = 0
            self.packet_count = 0
            self.angle_packet_count = 0
            self.checksum_error_count = 0
            self.packet_type_counts.clear()
            self.raw_preview.clear()

    def stats(self) -> dict[str, object]:
        """Return debug counters for the serial stream."""

        with self._lock:
            return {
                "raw_byte_count": self.raw_byte_count,
                "packet_count": self.packet_count,
                "angle_packet_count": self.angle_packet_count,
                "checksum_error_count": self.checksum_error_count,
                "sample_count": len(self.samples),
                "packet_type_counts": dict(self.packet_type_counts),
                "raw_preview_hex": self.raw_preview.hex(" "),
            }

    def to_frame(self, start_timestamp_s: float | None = None) -> pd.DataFrame:
        """Return collected samples as a DataFrame.

        If `start_timestamp_s` is provided, timestamps are made relative to that
        video timestamp. This is useful for aligning with `time_from_start_s`.
        """

        with self._lock:
            samples = list(self.samples)
        rows = []
        for sample in samples:
            timestamp = sample.timestamp_s
            if start_timestamp_s is not None:
                timestamp -= start_timestamp_s
            rows.append(
                {
                    "timestamp_s": timestamp,
                    "pitch_deg": sample.pitch_deg,
                    "roll_deg": sample.roll_deg,
                    "yaw_deg": sample.yaw_deg,
                    "accel_x": sample.accel_x,
                    "accel_y": sample.accel_y,
                    "accel_z": sample.accel_z,
                    "gyro_x": sample.gyro_x,
                    "gyro_y": sample.gyro_y,
                    "gyro_z": sample.gyro_z,
                }
            )
        return pd.DataFrame(rows)

    def to_series(self, start_timestamp_s: float | None = None) -> ImuOrientationSeries | None:
        """Return collected samples as an interpolable series."""

        frame = self.to_frame(start_timestamp_s=start_timestamp_s)
        if frame.empty:
            return None
        return ImuOrientationSeries(
            timestamp_s=frame["timestamp_s"].to_numpy(dtype=float),
            pitch_deg=frame["pitch_deg"].to_numpy(dtype=float),
            roll_deg=frame["roll_deg"].to_numpy(dtype=float),
            yaw_deg=frame["yaw_deg"].to_numpy(dtype=float),
        )

    def save_csv(self, path: str | Path, start_timestamp_s: float | None = None) -> None:
        """Save collected samples to CSV."""

        self.to_frame(start_timestamp_s=start_timestamp_s).to_csv(path, index=False)

    def _witmotion_read_loop(self) -> None:
        """Continuously poll the official `witmotion` Python interface."""

        last_angle = None
        while not self._stop_event.is_set():
            if self._imu is None:
                break
            angle = self._imu.get_angle()
            accel = self._imu.get_acceleration()
            gyro = self._imu.get_angular_velocity()
            if (
                (not angle or any(value is None for value in angle))
                and accel is None
                and gyro is None
            ):
                time.sleep(self.timeout_s)
                continue

            current_state = (angle, accel, gyro)
            if current_state == last_angle:
                time.sleep(self.timeout_s)
                continue
            last_angle = current_state

            if angle and not any(value is None for value in angle):
                roll, pitch, yaw = angle
            else:
                roll, pitch, yaw = None, None, None
            sample = Bwt901clSample(
                timestamp_s=time.monotonic(),
                pitch_deg=float(pitch) if pitch is not None else None,
                roll_deg=float(roll) if roll is not None else None,
                yaw_deg=float(yaw) if yaw is not None else None,
                accel_x=float(accel[0]) if accel else None,
                accel_y=float(accel[1]) if accel else None,
                accel_z=float(accel[2]) if accel else None,
                gyro_x=float(gyro[0]) if gyro else None,
                gyro_y=float(gyro[1]) if gyro else None,
                gyro_z=float(gyro[2]) if gyro else None,
            )
            with self._lock:
                self.raw_byte_count += 1
                self.packet_count += 1
                if pitch is not None and roll is not None and yaw is not None:
                    self.angle_packet_count += 1
                self.samples.append(sample)
            time.sleep(self.timeout_s)

    def _manual_read_loop(self) -> None:
        """Continuously read and decode WIT packets manually."""

        packet = bytearray()
        while not self._stop_event.is_set():
            if self._serial is None:
                break
            byte = self._serial.read(1)
            if not byte:
                continue
            value = byte[0]
            with self._lock:
                self.raw_byte_count += 1
                if len(self.raw_preview) < 80:
                    self.raw_preview.append(value)
            if not packet:
                if value != 0x55:
                    continue
                packet.append(value)
                continue

            packet.append(value)
            if len(packet) == 2 and packet[1] not in {0x51, 0x52, 0x53, 0x54, 0x56}:
                packet.clear()
                continue
            if len(packet) < 11:
                continue

            raw = bytes(packet[:11])
            del packet[:11]
            with self._lock:
                self.packet_count += 1
                self.packet_type_counts[raw[1]] = self.packet_type_counts.get(raw[1], 0) + 1
                if (sum(raw[:10]) & 0xFF) != raw[10]:
                    self.checksum_error_count += 1
                elif raw[1] == ANGLE_PACKET_ID:
                    self.angle_packet_count += 1
            sample = decode_angle_packet(raw, timestamp_s=time.monotonic())
            if sample is not None:
                with self._lock:
                    self.samples.append(sample)


def decode_angle_packet(packet: bytes, timestamp_s: float | None = None) -> Bwt901clSample | None:
    """Decode a WIT angle packet, returning None for other packet types."""

    if len(packet) != 11 or packet[0] != 0x55:
        return None
    if (sum(packet[:10]) & 0xFF) != packet[10]:
        return None
    if packet[1] != ANGLE_PACKET_ID:
        return None

    roll_raw = _signed_int16(packet[2], packet[3])
    pitch_raw = _signed_int16(packet[4], packet[5])
    yaw_raw = _signed_int16(packet[6], packet[7])
    return Bwt901clSample(
        timestamp_s=time.monotonic() if timestamp_s is None else timestamp_s,
        pitch_deg=pitch_raw / 32768.0 * 180.0,
        roll_deg=roll_raw / 32768.0 * 180.0,
        yaw_deg=yaw_raw / 32768.0 * 180.0,
    )


def _signed_int16(low: int, high: int) -> int:
    """Convert little-endian bytes to signed int16."""

    value = (high << 8) | low
    if value >= 32768:
        value -= 65536
    return value


def configure_bwt901cl_angle_output(serial_port) -> None:
    """Ask the sensor to output angle packets.

    This writes standard WIT register commands: unlock, enable angle output, set
    a moderate output rate, save. Some sensors already stream by default, but
    this helps when a port opens cleanly and no bytes arrive.
    """

    for command in (UNLOCK_COMMAND, ANGLE_OUTPUT_COMMAND, RATE_10HZ_COMMAND, SAVE_COMMAND):
        serial_port.write(command)
        serial_port.flush()
        time.sleep(0.05)


def list_serial_ports() -> list[str]:
    """List serial ports visible to the current machine."""

    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is required. Install it with `pip install pyserial`.") from exc
    return [port.device for port in list_ports.comports()]
