"""
Stream endpoint: accepts a single base64-encoded frame from the live camera
or manages background CCTV RTSP pulling.
"""
import asyncio
import base64
import uuid
import cv2
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis, get_redis_pool
from app.db.models import LifecycleEvent, Report, ReportStatus
from app.db.session import get_db, AsyncSessionLocal
from app.schemas.report import PerceptionResult
from app.services.deduplication import is_duplicate
from app.services.image_store import save_image
from app.services.qwen_client import call_qwen_vision

logger = get_logger(__name__)
router = APIRouter(prefix="/stream", tags=["stream"])

# Shared state for background CCTV tasks
active_cctv_tasks = {}


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


class FrameResponse(BaseModel):
    action: str              # "reported" | "duplicate" | "no_issue" | "low_confidence"
    report_id: str | None = None
    existing_report_id: str | None = None
    issue_type: str | None = None
    confidence: float | None = None
    message: str = ""


async def process_frame_logic(
    image_bytes: bytes,
    db: AsyncSession,
    redis: aioredis.Redis,
    lat: float | None,
    lon: float | None,
    camera_label: str,
) -> FrameResponse:
    """Core logic to process an image, check duplicates, and create report."""
    settings = get_settings()

    # 1. Run perception (Qwen3-VL)
    try:
        qwen = await call_qwen_vision(image_bytes, "image/jpeg")
    except RuntimeError as exc:
        logger.error("perception_failed", error=str(exc))
        return FrameResponse(action="error", message="Perception service unavailable.")

    # 2. No issue detected
    if not qwen.issues or qwen.confidence_score < settings.vision_confidence_threshold:
        return FrameResponse(
            action="no_issue" if not qwen.issues else "low_confidence",
            confidence=qwen.confidence_score,
            message="No civic issue detected.",
        )

    primary_issue = max(qwen.issues, key=lambda i: i.severity)

    # 3. Deduplication check
    is_dup, existing_id = await is_duplicate(db, primary_issue.type, lat, lon)

    if is_dup:
        logger.info("duplicate_skipped", existing=existing_id, label=camera_label)
        return FrameResponse(
            action="duplicate",
            existing_report_id=existing_id,
            issue_type=primary_issue.type,
            confidence=qwen.confidence_score,
            message=f"Already reported nearby.",
        )

    # 4. New issue — create report
    report = Report(
        original_filename=f"stream_{camera_label}_{uuid.uuid4().hex[:8]}.jpg",
        status=ReportStatus.RECEIVED,
        gps_latitude=lat,
        gps_longitude=lon,
        confidence_score=qwen.confidence_score,
    )
    db.add(report)
    await db.flush()

    db.add(LifecycleEvent(
        report_id=report.id,
        from_status=None,
        to_status=ReportStatus.RECEIVED,
        detail=f"Auto-detected from stream ({camera_label}): {primary_issue.type}",
    ))

    save_image(report.id, image_bytes, "image/jpeg")

    from app.schemas.report import DetectedIssue, PerceptionResult as PR
    perception = PR(
        report_id=report.id,
        summary=qwen.summary,
        confidence_score=qwen.confidence_score,
        issues=[
            DetectedIssue(
                type=i.type,
                bbox_ymin=i.bbox[0], bbox_xmin=i.bbox[1],
                bbox_ymax=i.bbox[2], bbox_xmax=i.bbox[3],
                severity=i.severity,
                description=i.description,
            ) for i in qwen.issues
        ],
        gps_latitude=lat,
        gps_longitude=lon,
    )
    report.perception_result = perception.model_dump(mode="json")
    report.status = ReportStatus.ANALYZED

    db.add(LifecycleEvent(
        report_id=report.id,
        from_status=ReportStatus.RECEIVED,
        to_status=ReportStatus.ANALYZED,
        detail="Perception complete (fast-path)",
    ))

    await db.commit()
    await redis.rpush(settings.knowledge_queue_key, str(report.id))

    logger.info("report_created", report_id=str(report.id), issue=primary_issue.type)
    return FrameResponse(
        action="reported",
        report_id=str(report.id),
        issue_type=primary_issue.type,
        confidence=qwen.confidence_score,
        message=f"New {primary_issue.type} reported.",
    )


@router.post("/frame", response_model=FrameResponse)
async def analyse_frame(
    payload: FramePayload,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
):
    # Decode base64
    try:
        b64_data = payload.image_b64.split(",")[1] if "," in payload.image_b64 else payload.image_b64
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    return await process_frame_logic(
        image_bytes, db, redis, payload.latitude, payload.longitude, payload.camera_label
    )


async def cctv_worker_task(rtsp_url: str, lat: float | None, lon: float | None, label: str):
    """Loop to pull frames from CCTV."""
    logger.info("cctv_worker_starting", url=rtsp_url, label=label)
    
    cap = cv2.VideoCapture(rtsp_url)
    try:
        while rtsp_url in active_cctv_tasks:
            ret, frame = cap.read()
            if not ret:
                logger.warning("cctv_frame_failed", url=rtsp_url)
                cap.release()
                await asyncio.sleep(10)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Every 15 seconds
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                image_bytes = buffer.tobytes()
                # Run logic in a new session
                async with AsyncSessionLocal() as db:
                    pool = get_redis_pool()
                    async with aioredis.Redis(connection_pool=pool) as redis:
                        await process_frame_logic(image_bytes, db, redis, lat, lon, label)

            await asyncio.sleep(15)
    except Exception as e:
        logger.error("cctv_worker_error", url=rtsp_url, error=str(e))
    finally:
        cap.release()
        active_cctv_tasks.pop(rtsp_url, None)
        logger.info("cctv_worker_stopped", url=rtsp_url)


@router.post("/cctv/start")
async def start_cctv(payload: CCTVStartPayload):
    if payload.rtsp_url in active_cctv_tasks:
        return {"status": "already_running", "url": payload.rtsp_url}

    task = asyncio.create_task(
        cctv_worker_task(payload.rtsp_url, payload.latitude, payload.longitude, payload.label)
    )
    active_cctv_tasks[payload.rtsp_url] = task
    return {"status": "started", "url": payload.rtsp_url}


@router.post("/cctv/stop")
async def stop_cctv(rtsp_url: str):
    if rtsp_url in active_cctv_tasks:
        # Removal from dict triggers loop exit
        active_cctv_tasks.pop(rtsp_url)
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.get("/cctv/status")
async def get_cctv_status():
    return {"active_streams": list(active_cctv_tasks.keys())}
