import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis
from app.db.session import get_db
from app.db.models import DeadLetterQueue, LifecycleEvent, Report, ReportStatus
from app.schemas.report import IncomingReport, ReportResponse

logger=get_logger(__name__)
router=APIRouter(prefix="/reports", tags=["reports"])

ALLOWED_CONTENT_TYPES={"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES=15*1024*1024

async def _check_rate_limit(client_id: str, redis: aioredis.Redis) -> None:
    settings = get_settings()
    key = f"ratelimit:{client_id}"
    count= await redis.incr(key)
    if count == 1:
        await redis.expire(key, settings.rate_limit_window_seconds)
    if count>settings.rate_limiit_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before submitting another report"
        )

@router.post("/", response_model=ReportResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_report(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis)
    ):

    await _check_rate_limit(client_id="global", redis=redis)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Only JPEG, PNG, and WebP images are allowed."
        )

    try:
        IncomingReport(
            filename=file.filename or "unknown",
            content_type=file.content_type     
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_UNPROCESSABLE_ENTITY, detail=str(exc))

    contents=await file.read()
    if len(contents)>MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is too large. Maximum size is 15MB."
        )

    report=Report(
        original_filename=file.filename or "unknown",
        status=ReportStatus.RECEIVED
    )
    db.add(report)
    await db.flush()

    event = LifecycleEvent(
        report_id=report.id,
        to_status=ReportStatus.RECEIVED,
        detail="Report received and validated",
    )
    db.add(event)
    await db.commit()
    await db.refresh(report)

    logger.info("report_received", report_id=str(report.id), filename=file.filename)

    return ReportResponse(
        id=report.id,
        status=report.status.value,
        message="Report received. It will be processed and filed shortly.",
        created_at=report.created_at,
    )
@router.get("/{report_id}", response_model=ReportResponse)
async def get_report_status(
    report_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
    ):

    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )
    return ReportResponse(
        id=report.id,
        status=report.status.value,
        message=f"Report is currently {report.status.value.lower()}.",
        created_at=report.created_at,
    ) 
