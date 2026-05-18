from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from jump_analysis.service import VideoAnalysisService

app = FastAPI(title="Jump Analysis API", version="0.1.0")
service = VideoAnalysisService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-jump")
async def analyze_jump(
    file: UploadFile = File(...),
    height_cm: float = Form(...),
) -> dict[str, object]:
    suffix = Path(file.filename or "upload.mov").suffix or ".mov"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(await file.read())
            temp_path = Path(temporary_file.name)

        result = service.analyze_video(temp_path, height_cm=height_cm)
        return result.as_json()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
