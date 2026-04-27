import os
import uuid
from pathlib import Path
from app.core.logging import get_logger

logger=get_logger(__name__)

_BASE_PATH=Path(os.getenv("IMAGE_STORE_PATH", "/tmp/images"))
_MIME_TO_EXT={
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

def save_image(report_id: uuid.UUID, image_bytes: bytes, mime_type: str) -> str:
    _BASE_PATH.mkdir(parents=True, exist_ok=True)
    ext = _MIME_TO_EXT.get(mime_type, ".jpg")
    path = _BASE_PATH / f"{report_id}{ext}"
    path.write_bytes(image_bytes)
    logger.info("image_saved", report_id=str(report_id), path=str(path))
    return str(path)

def load_image(report_id: uuid.UUID) -> tuple[bytes, str] | None:
    for mime, ext in _MIME_TO_EXT.items():
        path=_BASE_PATH / f"{report_id}{ext}"
        if path.exists():
            return path.read_bytes(), mime
    return None