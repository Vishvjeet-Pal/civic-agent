"""
Stream service: core logic for processing image frames from camera or CCTV.
"""
import uuid
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import LifecycleEvent, Report, ReportStatus, ReportSubscriber, User, AuthorityStatus
from app.schemas.report import PerceptionResult, DetectedIssue
from app.services.deduplication import is_duplicate
from app.services.image_store import save_image
from app.services.qwen_client import call_qwen_vision
from pydantic import BaseModel

logger = get_logger(__name__)

class FrameResponse(BaseModel):
    action: str              # "reported" | "duplicate" | "no_issue" | "low_confidence" | "error"
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
    user: User | None = None  # Added user context
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
    is_dup, existing_id = await is_duplicate(db, primary_issue.type, lat, lon, user)

    if is_dup:
        logger.info("duplicate_subscriber_added", existing=existing_id, label=camera_label)
        # Commit the subscriber added in is_duplicate
        await db.commit()
        return FrameResponse(
            action="duplicate",
            existing_report_id=existing_id,
            issue_type=primary_issue.type,
            confidence=qwen.confidence_score,
            message=f"Already reported nearby. You've been added to notifications.",
        )

    # 4. New issue — create report
    report = Report(
        original_filename=f"stream_{camera_label}_{uuid.uuid4().hex[:8]}.jpg",
        status=ReportStatus.RECEIVED,
        authority_status=AuthorityStatus.PENDING,
        gps_latitude=lat,
        gps_longitude=lon,
        confidence_score=qwen.confidence_score,
        reporter_id=user.id if user else None
    )
    db.add(report)
    await db.flush()

    # Add initial subscriber
    if user:
        db.add(ReportSubscriber(report_id=report.id, user_id=user.id))

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
