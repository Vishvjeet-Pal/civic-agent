"""
Stream endpoint: accepts a single base64-encoded frame from the live camera
or manages background CCTV RTSP pulling.
"""
import base64
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.core.redis import get_redis
from app.db.session import get_db
from app.db.models import User
from app.services.auth import get_current_user, get_current_authority
from app.services.stream_service import process_frame_logic, FrameResponse
from app.workers.cctv_worker import start_cctv_task, stop_cctv_task, active_cctv_tasks

router = APIRouter(prefix="/stream", tags=["stream"])


class FramePayload(BaseModel):
    """Single frame submitted from the browser live stream."""
    image_b64: str           # base64-encoded JPEG
    latitude: float | None = None
    longitude: float | None = None
    camera_label: str = "mobile"


class CCTVStartPayload(BaseModel):
    rtsp_url: str
    latitude: float | None = None
    longitude: float | None = None
    label: str = "cctv"


@router.post("/frame", response_model=FrameResponse)
async def analyse_frame(
    payload: FramePayload,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user)
):
    # Decode base64
    try:
        b64_data = payload.image_b64.split(",")[1] if "," in payload.image_b64 else payload.image_b64
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    return await process_frame_logic(
        image_bytes, db, redis, payload.latitude, payload.longitude, payload.camera_label, current_user
    )


@router.post("/cctv/start")
async def start_cctv(
    payload: CCTVStartPayload,
    current_user: User = Depends(get_current_user)
):
    # For CCTV, we might not always have a user context in the background worker,
    # but the START command should definitely be authenticated.
    started = start_cctv_task(payload.rtsp_url, payload.latitude, payload.longitude, payload.label)
    if not started:
        return {"status": "already_running", "url": payload.rtsp_url}
    return {"status": "started", "url": payload.rtsp_url}


@router.post("/cctv/stop")
async def stop_cctv(
    rtsp_url: str,
    current_user: User = Depends(get_current_user)
):
    stopped = stop_cctv_task(rtsp_url)
    if stopped:
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/cctv/status")
async def get_cctv_status(current_user: User = Depends(get_current_user)):
    return {"active_streams": list(active_cctv_tasks.keys())}
