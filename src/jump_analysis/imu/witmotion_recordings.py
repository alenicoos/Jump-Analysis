from __future__ import annotations

import csv
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("jump_analysis.imu")


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "null":
        return None
    return float(cleaned)


def _parse_local_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone()


@dataclass(slots=True)
class IMUDeviceSummary:
    device_name: str
    sample_count: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    mean_acceleration_g: float | None
    peak_acceleration_g: float | None
    mean_angular_velocity_dps: float | None
    peak_angular_velocity_dps: float | None
    mean_roll_deg: float | None
    mean_pitch_deg: float | None
    mean_yaw_deg: float | None
    mean_temperature_c: float | None
    battery_start_percent: int | None
    battery_end_percent: int | None

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["start_time"] = self.start_time.isoformat()
        payload["end_time"] = self.end_time.isoformat()
        return payload


@dataclass(slots=True)
class IMURecordingSummary:
    matched_file: str
    matched_folder: str
    recording_start_time: datetime
    recording_end_time: datetime
    time_offset_seconds: float
    device_count: int
    total_samples: int
    device_summaries: list[IMUDeviceSummary]

    def as_json(self) -> dict[str, object]:
        return {
            "matched_file": self.matched_file,
            "matched_folder": self.matched_folder,
            "recording_start_time": self.recording_start_time.isoformat(),
            "recording_end_time": self.recording_end_time.isoformat(),
            "time_offset_seconds": self.time_offset_seconds,
            "device_count": self.device_count,
            "total_samples": self.total_samples,
            "device_summaries": [summary.as_json() for summary in self.device_summaries],
        }

    def short_summary(self) -> str:
        if not self.device_summaries:
            return "WitMotion recording found, but it did not contain valid IMU device samples."

        device_bits = []
        for summary in self.device_summaries:
            peak_accel = f"{summary.peak_acceleration_g:.2f} g" if summary.peak_acceleration_g is not None else "n/a"
            peak_gyro = (
                f"{summary.peak_angular_velocity_dps:.1f} deg/s"
                if summary.peak_angular_velocity_dps is not None
                else "n/a"
            )
            device_bits.append(
                f"{summary.device_name}: {summary.sample_count} samples, peak accel {peak_accel}, peak gyro {peak_gyro}"
            )

        return (
            f"WitMotion recording matched with {self.device_count} sensor(s) over "
            f"{(self.recording_end_time - self.recording_start_time).total_seconds():.2f} s. "
            + " ".join(device_bits)
        )


@dataclass(slots=True)
class _RecordingCandidate:
    txt_path: Path
    start_time: datetime
    end_time: datetime
    row_count: int


class WitMotionRecordingFinder:
    def __init__(
        self,
        recordings_root: str | Path | None = None,
        max_time_offset: timedelta = timedelta(minutes=20),
    ) -> None:
        self.recordings_root = (
            Path(recordings_root).expanduser()
            if recordings_root is not None
            else Path.home() / "Library/Containers/com.witmotion.app/Data/Documents/WitMotionRecord"
        )
        self.max_time_offset = max_time_offset

    def find_matching_recording(self, reference_time: datetime) -> IMURecordingSummary | None:
        candidates = self._scan_candidates()
        if not candidates:
            logger.info(
                "IMU match: no .txt recordings found under root=%s for reference_time=%s",
                self.recordings_root,
                reference_time.astimezone().isoformat(),
            )
            return None

        reference_time = reference_time.astimezone()
        logger.info(
            "IMU match: evaluating %s candidate .txt file(s) under root=%s for reference_time=%s",
            len(candidates),
            self.recordings_root,
            reference_time.isoformat(),
        )

        def score(candidate: _RecordingCandidate) -> tuple[float, float]:
            if candidate.start_time <= reference_time <= candidate.end_time:
                primary = 0.0
            elif reference_time < candidate.start_time:
                primary = (candidate.start_time - reference_time).total_seconds()
            else:
                primary = (reference_time - candidate.end_time).total_seconds()
            midpoint = candidate.start_time + (candidate.end_time - candidate.start_time) / 2
            secondary = abs((midpoint - reference_time).total_seconds())
            return (primary, secondary)

        ranked = sorted(candidates, key=score)
        best = ranked[0]
        best_offset = score(best)[0]
        if best_offset > self.max_time_offset.total_seconds():
            logger.info(
                "IMU match: closest file=%s offset_seconds=%.3f exceeds max_time_offset_seconds=%.3f",
                best.txt_path,
                best_offset,
                self.max_time_offset.total_seconds(),
            )
            return None

        logger.info(
            "IMU match: selected file=%s window_start=%s window_end=%s row_count=%s offset_seconds=%.3f",
            best.txt_path,
            best.start_time.isoformat(),
            best.end_time.isoformat(),
            best.row_count,
            best_offset,
        )
        return self._parse_recording(best, reference_time)

    def _scan_candidates(self) -> list[_RecordingCandidate]:
        if not self.recordings_root.exists():
            return []

        candidates: list[_RecordingCandidate] = []
        for txt_path in sorted(self.recordings_root.rglob("*.txt")):
            try:
                candidate = self._summarize_candidate(txt_path)
            except Exception:
                continue
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _summarize_candidate(self, txt_path: Path) -> _RecordingCandidate | None:
        start_time: datetime | None = None
        end_time: datetime | None = None
        row_count = 0

        with txt_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                raw_time = row.get("time")
                if not raw_time:
                    continue
                row_time = _parse_local_iso8601(raw_time)
                start_time = row_time if start_time is None else min(start_time, row_time)
                end_time = row_time if end_time is None else max(end_time, row_time)
                row_count += 1

        if start_time is None or end_time is None or row_count == 0:
            return None

        return _RecordingCandidate(
            txt_path=txt_path,
            start_time=start_time,
            end_time=end_time,
            row_count=row_count,
        )

    def _parse_recording(self, candidate: _RecordingCandidate, reference_time: datetime) -> IMURecordingSummary:
        per_device: dict[str, dict[str, object]] = {}

        with candidate.txt_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                device_name = (row.get("DeviceName") or "").strip()
                raw_time = row.get("time")
                if not device_name or not raw_time:
                    continue

                timestamp = _parse_local_iso8601(raw_time)
                device_bucket = per_device.setdefault(
                    device_name,
                    {
                        "times": [],
                        "accel": [],
                        "gyro": [],
                        "roll": [],
                        "pitch": [],
                        "yaw": [],
                        "temperature": [],
                        "battery": [],
                    },
                )
                device_bucket["times"].append(timestamp)

                ax = _safe_float(row.get("AccX(g)"))
                ay = _safe_float(row.get("AccY(g)"))
                az = _safe_float(row.get("AccZ(g)"))
                if ax is not None and ay is not None and az is not None:
                    device_bucket["accel"].append(math.sqrt(ax * ax + ay * ay + az * az))

                gx = _safe_float(row.get("AsX(°/s)"))
                gy = _safe_float(row.get("AsY(°/s)"))
                gz = _safe_float(row.get("AsZ(°/s)"))
                if gx is not None and gy is not None and gz is not None:
                    device_bucket["gyro"].append(math.sqrt(gx * gx + gy * gy + gz * gz))

                for source_key, target_key in (
                    ("AngleX(°)", "roll"),
                    ("AngleY(°)", "pitch"),
                    ("AngleZ(°)", "yaw"),
                    ("Temperature(°C)", "temperature"),
                ):
                    value = _safe_float(row.get(source_key))
                    if value is not None:
                        device_bucket[target_key].append(value)

                battery_value = _safe_float(row.get("Battery level(%)"))
                if battery_value is not None:
                    device_bucket["battery"].append(int(round(battery_value)))

        device_summaries: list[IMUDeviceSummary] = []
        total_samples = 0
        for device_name, bucket in sorted(per_device.items()):
            times: list[datetime] = bucket["times"]  # type: ignore[assignment]
            if not times:
                continue

            def mean_or_none(values: list[float]) -> float | None:
                return sum(values) / len(values) if values else None

            def max_or_none(values: list[float]) -> float | None:
                return max(values) if values else None

            accel_values: list[float] = bucket["accel"]  # type: ignore[assignment]
            gyro_values: list[float] = bucket["gyro"]  # type: ignore[assignment]
            roll_values: list[float] = bucket["roll"]  # type: ignore[assignment]
            pitch_values: list[float] = bucket["pitch"]  # type: ignore[assignment]
            yaw_values: list[float] = bucket["yaw"]  # type: ignore[assignment]
            temperature_values: list[float] = bucket["temperature"]  # type: ignore[assignment]
            battery_values: list[int] = bucket["battery"]  # type: ignore[assignment]

            summary = IMUDeviceSummary(
                device_name=device_name,
                sample_count=len(times),
                start_time=times[0],
                end_time=times[-1],
                duration_seconds=(times[-1] - times[0]).total_seconds(),
                mean_acceleration_g=mean_or_none(accel_values),
                peak_acceleration_g=max_or_none(accel_values),
                mean_angular_velocity_dps=mean_or_none(gyro_values),
                peak_angular_velocity_dps=max_or_none(gyro_values),
                mean_roll_deg=mean_or_none(roll_values),
                mean_pitch_deg=mean_or_none(pitch_values),
                mean_yaw_deg=mean_or_none(yaw_values),
                mean_temperature_c=mean_or_none(temperature_values),
                battery_start_percent=battery_values[0] if battery_values else None,
                battery_end_percent=battery_values[-1] if battery_values else None,
            )
            total_samples += summary.sample_count
            device_summaries.append(summary)

        return IMURecordingSummary(
            matched_file=str(candidate.txt_path),
            matched_folder=str(candidate.txt_path.parent),
            recording_start_time=candidate.start_time,
            recording_end_time=candidate.end_time,
            time_offset_seconds=self._compute_offset_seconds(candidate, reference_time),
            device_count=len(device_summaries),
            total_samples=total_samples,
            device_summaries=device_summaries,
        )

    def _compute_offset_seconds(self, candidate: _RecordingCandidate, reference_time: datetime) -> float:
        if candidate.start_time <= reference_time <= candidate.end_time:
            return 0.0
        if reference_time < candidate.start_time:
            return (candidate.start_time - reference_time).total_seconds()
        return (reference_time - candidate.end_time).total_seconds()
