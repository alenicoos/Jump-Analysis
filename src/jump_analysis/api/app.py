from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from jump_analysis.live_session import LiveGuidanceProcessor
from jump_analysis.service import VideoAnalysisService


def _configure_jump_analysis_logging() -> None:
    formatter = logging.Formatter(
        "%(levelname)s:%(name)s:%(message)s"
    )
    jump_logger = logging.getLogger("jump_analysis")
    if not any(getattr(handler, "_jump_analysis_handler", False) for handler in jump_logger.handlers):
        handler = logging.StreamHandler()
        handler._jump_analysis_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(formatter)
        jump_logger.addHandler(handler)
    jump_logger.setLevel(logging.INFO)
    jump_logger.propagate = False


_configure_jump_analysis_logging()

app = FastAPI(title="Jump Analysis API", version="0.1.0")
service = VideoAnalysisService()
logger = logging.getLogger("jump_analysis.api")
logger.setLevel(logging.INFO)


def _log_live_event(event: dict[str, object]) -> None:
    logger.info(
        "Live event sending type=%s phase=%s level=%s speak=%s text=%s",
        event.get("type"),
        event.get("phase"),
        event.get("level"),
        event.get("speak"),
        str(event.get("text", ""))[:160],
    )


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


@app.websocket("/live-session")
async def live_session(websocket: WebSocket) -> None:
    await websocket.accept()

    raw_height_cm = websocket.query_params.get("height_cm")
    client_host = getattr(websocket.client, "host", "unknown")
    client_port = getattr(websocket.client, "port", "unknown")
    try:
        height_cm = float(raw_height_cm) if raw_height_cm is not None else 0.0
    except ValueError as error:
        event = {
            "type": "error",
            "phase": "initializing",
            "text": "height_cm must be a valid number.",
            "level": "error",
            "speak": True,
        }
        _log_live_event(event)
        await websocket.send_json(event)
        raise ValueError("height_cm must be a valid number.") from error

    logger.info(
        "Live websocket connected client=%s:%s height_cm=%.2f",
        client_host,
        client_port,
        height_cm,
    )

    if height_cm <= 0:
        event = {
            "type": "error",
            "phase": "initializing",
            "text": "A positive athlete height is required before starting live guidance.",
            "level": "error",
            "speak": True,
        }
        _log_live_event(event)
        await websocket.send_json(event)
        await websocket.close(code=1008)
        return

    processor = LiveGuidanceProcessor(
        analysis_service=service,
        model=service._get_model(),
        height_cm=height_cm,
    )
    connected_event = {
        "type": "status",
        "phase": "floor_setup",
        "text": "Live guidance connected. Stand on the floor facing the camera and stay still.",
        "level": "info",
        "speak": True,
    }
    _log_live_event(connected_event)
    await websocket.send_json(connected_event)

    try:
        while True:
            message = await websocket.receive_text()
            logger.info(
                "Live websocket message received phase=%s size_bytes=%s",
                processor.phase,
                len(message.encode("utf-8")),
            )
            for event in processor.process_text_message(message):
                _log_live_event(event)
                await websocket.send_json(event)
            for event in processor.consume_pending_analysis():
                _log_live_event(event)
                await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info(
            "Live guidance websocket disconnected client=%s:%s final_phase=%s",
            client_host,
            client_port,
            processor.phase,
        )
