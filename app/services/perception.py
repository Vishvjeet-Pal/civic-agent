"""
Perception service: coordinates EXIF extraction, Qwen3-VL call,
confidence routing, DB persistence, and queue handoff.
"""
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import DeadLetterQueue, LifecycleEvent, Report, ReportStatus
from app.schemas.report import BoundingBox, PerceptionResult
from app.services.exif import extract_exif
from app.services.qwen_client import QwenResponse, call_qwen_vision

logger = get_logger(__name__)

async def _transition_status(db: AsyncSession, report: Report, to_status: ReportStatus, detail: str) -> None:
    from_status=report.status
    report.status=to_status
    db.add(
        LifecycleEvent(
            report_id=report.id,
            from_status=from_status,
            to_status=to_status,
            detail=detail,
        )
    )
    await db.flush()

async def run_perception(report_id: uuid.UUID, image_bytes: bytes, mime_type: str, db: AsyncSession, redis: aioredis.Redis) -> PerceptionResult| None:
    """
    Full perception pipeline for a single report.
    Returns PerceptionResult on success, None if routed to human review or DLQ.
    """
    settings = get_settings()
    report = await db.get(Report, report_id)

    if not report:
        logger.error("perception_report_not_found", report_id=str(report_id))
        return None

    await _transition_status(db, report, ReportStatus.PROCESSING, "Perception started")
    await db.commit()

    # ── 1. EXIF extraction ─────────────────────────────────────────────────
    exif = extract_exif(image_bytes)
    logger.info(
        "exif_extracted",
        report_id=str(report_id),
        has_gps=exif.latitude is not None,
        has_timestamp=exif.captured_at is not None,
    )

    # ── 2. Qwen3-VL call (with retry inside the client) ────────────────────
    try:
        qwen_result: QwenResponse = await call_qwen_vision(image_bytes, mime_type)
    except RuntimeError as exc:
        # All retries exhausted → dead-letter queue
        logger.error("perception_qwen_exhausted", report_id=str(report_id), error=str(exc))
        db.add(
            DeadLetterQueue(
                report_id=report_id,
                phase="perception",
                error_detail=str(exc),
                retry_count=settings.qwen_max_retries,
            )
        )
        await _transition_status(db, report, ReportStatus.FAILED, f"Qwen3-VL failed: {exc}")
        await db.commit()
        return None

    # ── 3. Confidence routing ──────────────────────────────────────────────
    threshold = settings.vision_confidence_threshold
    low_confidence = qwen_result.confidence_score < threshold

    if low_confidence:
        logger.info(
            "perception_low_confidence",
            report_id=str(report_id),
            score=qwen_result.confidence_score,
            threshold=threshold,
        )
        # Persist partial data and route to human review
        report.gps_latitude = exif.latitude
        report.gps_longitude = exif.longitude
        report.captured_at = exif.captured_at
        report.confidence_score = qwen_result.confidence_score
        report.perception_result = qwen_result.model_dump()
        await _transition_status(
            db,
            report,
            ReportStatus.PENDING_REVIEW,
            f"Confidence {qwen_result.confidence_score:.2f} below threshold {threshold}",
        )
        await db.commit()
        return None

    # ── 4. Build validated PerceptionResult ────────────────────────────────
    bbox = None
    if qwen_result.bounding_box:
        bbox = BoundingBox(**qwen_result.bounding_box)

    result = PerceptionResult(
        report_id=report_id,
        issue_detected=qwen_result.issue_detected,
        issue_label=qwen_result.issue_label,
        confidence_score=qwen_result.confidence_score,
        bounding_box=bbox,
        gps_latitude=exif.latitude,
        gps_longitude=exif.longitude,
        captured_at=exif.captured_at,
        low_confidence=False,
    )

    # ── 5. Persist perception output ───────────────────────────────────────
    report.gps_latitude = result.gps_latitude
    report.gps_longitude = result.gps_longitude
    report.captured_at = result.captured_at
    report.confidence_score = result.confidence_score
    report.perception_result = result.model_dump(mode="json")
    await _transition_status(db, report, ReportStatus.ANALYZED, "Perception complete")
    await db.commit()

    logger.info(
        "perception_complete",
        report_id=str(report_id),
        issue_label=result.issue_label,
        confidence=result.confidence_score,
    )

    # ── 6. Enqueue for Phase 3 (knowledge) ────────────────────────────────
    await redis.rpush(settings.knowledge_queue_key, str(report_id))

    return result