import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis
from app.db.session import get_db
from app.db.models import DeadLetterQueue, LifecycleEvent, Report, ReportStatus, ReportSubscriber, User, UserRole, AuthorityStatus
from app.schemas.report import IncomingReport, ReportResponse, ReportDetail
from app.services.image_store import save_image, delete_image
from app.services.auth import get_current_user

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
    if count>settings.rate_limit_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before submitting another report"
        )

@router.post("/", response_model=ReportResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_report(
    file: UploadFile = File(...),
    address: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user)
    ):

    await _check_rate_limit(client_id=str(current_user.id), redis=redis)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Only JPEG, PNG, and WebP images are allowed."
        )

    try:
        IncomingReport(
            filename=file.filename or "unknown",
            content_type=file.content_type,
            address=address
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
        status=ReportStatus.RECEIVED,
        authority_status=AuthorityStatus.PENDING,
        provided_address=address,
        reporter_id=current_user.id
    )
    db.add(report)
    await db.flush()

    # Initial subscriber
    db.add(ReportSubscriber(report_id=report.id, user_id=current_user.id))

    event = LifecycleEvent(
        report_id=report.id,
        to_status=ReportStatus.RECEIVED,
        detail="Report received and validated",
    )
    db.add(event)
    await db.commit()
    await db.refresh(report)

    logger.info("report_received", report_id=str(report.id), filename=file.filename)

    try:
        saved_path = save_image(report.id, contents, file.content_type)
    except OSError as exc:
        logger.error("image_store_failed", report_id=str(report.id), error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist image to disk.",
        )

    await redis.rpush("perception:queue", str(report.id))

    return ReportResponse(
        id=report.id,
        status=report.status.value,
        message="Report received. It will be processed and filed shortly.",
        created_at=report.created_at,
    )

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report_status(
    report_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
    ):

    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")
    
    # Check if user is the reporter OR a subscriber
    sub_check = await db.execute(
        select(ReportSubscriber).where(
            and_(ReportSubscriber.report_id == report_id, ReportSubscriber.user_id == current_user.id)
        )
    )
    if report.reporter_id != current_user.id and not sub_check.scalar_one_or_none():
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    return ReportResponse(
        id=report.id,
        status=report.status.value,
        message=f"Report is currently {report.status.value.lower()}.",
        created_at=report.created_at,
    )

@router.get("/", response_model=list[ReportDetail])
async def list_reports(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
    ):
    """List reports visible to the current user (Citizen's own or Authority's dept)."""
    
    if current_user.role == UserRole.AUTHORITY:
        # Authorities see reports assigned to their department email
        # Use postgres JSONB contains operator
        query = select(Report).where(
            Report.assigned_department_emails.contains([current_user.email])
        )
    else:
        # Citizens see reports they created OR subscribed to
        query = select(Report).join(ReportSubscriber, Report.id == ReportSubscriber.report_id).where(
            ReportSubscriber.user_id == current_user.id
        )
    
    query = query.order_by(Report.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    reports = result.scalars().all()
    return reports

@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
    ):
    """Delete a report (only by the original reporter)."""
    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")
    
    if report.reporter_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the original reporter can delete.")
    
    await db.delete(report)
    await db.commit()
    delete_image(report_id)
    return None
