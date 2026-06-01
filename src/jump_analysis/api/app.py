from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from jump_analysis.service import VideoAnalysisService

app = FastAPI(title="Jump Analysis API", version="0.1.0")
service = VideoAnalysisService()
logger = logging.getLogger("jump_analysis.api")
logger.setLevel(logging.INFO)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-jump")
async def analyze_jump(
    file: UploadFile = File(...),
    height_cm: float = Form(...),
    recording_started_at: str | None = Form(default=None),
) -> dict[str, object]:
    suffix = Path(file.filename or "upload.mov").suffix or ".mov"
    temp_path: Path | None = None
    try:
        video_bytes = await file.read()
        received_at = datetime.now().astimezone()
        logger.info(
            "Request received: video arrived filename=%s size_bytes=%s height_cm=%.2f received_at=%s",
            file.filename or "upload.mov",
            len(video_bytes),
            height_cm,
            received_at.isoformat(),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(video_bytes)
            temp_path = Path(temporary_file.name)
        logger.info("Upload persisted temporarily temp_path=%s suffix=%s", temp_path, suffix)

        parsed_recording_started_at: datetime | None = None
        if recording_started_at is not None:
            try:
                parsed_recording_started_at = datetime.fromisoformat(recording_started_at)
                if parsed_recording_started_at.tzinfo is None:
                    parsed_recording_started_at = parsed_recording_started_at.astimezone()
            except ValueError as error:
                raise ValueError("recording_started_at must be a valid ISO 8601 datetime.") from error
        logger.info(
            "Request metadata: recording_started_at=%s",
            parsed_recording_started_at.astimezone().isoformat() if parsed_recording_started_at is not None else "none",
        )

        result = service.analyze_video_with_timestamp(
            temp_path,
            height_cm=height_cm,
            recording_started_at=parsed_recording_started_at,
        )
        payload = result.as_json()
        imu_recording = payload.get("imu_recording")
        logger.info(
            "IMU reference time used by server=%s",
            (
                parsed_recording_started_at.astimezone().isoformat()
                if parsed_recording_started_at is not None
                else received_at.isoformat()
            ),
        )
        if isinstance(imu_recording, dict):
            logger.info(
                "IMU selected for response matched_file=%s devices=%s total_samples=%s offset_seconds=%.3f",
                imu_recording.get("matched_file"),
                imu_recording.get("device_count"),
                imu_recording.get("total_samples"),
                float(imu_recording.get("time_offset_seconds", 0.0)),
            )
        else:
            logger.info("IMU selected for response: none")
        logger.info(
            "Sending response to app prediction=%s protocol_passed=%s anomaly_score=%.3f imu_attached=%s",
            payload.get("prediction"),
            payload.get("protocol_passed"),
            float(payload.get("anomaly_score", 0.0)),
            imu_recording is not None,
        )
        return payload
    except ValueError as error:
        logger.warning("Jump analysis rejected invalid input: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        logger.warning("Jump analysis failed validation: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Jump analysis crashed unexpectedly")
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        try:
            if temp_path is not None:
                logger.info("Cleaning up temporary upload file temp_path=%s", temp_path)
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass
