from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import DeadLetterQueue, Report, ReportStatus 
from app.db.session import get_db
from app.schemas.report import ReportDetail

router=APIRouter(prefix="/admin", tags=["admin"])

@router.get("/review-queue")
async def get_review_queue(
    limit: int = Query(default=50, le=200),
    db: AsyncSession=Depends(get_db)
):
    stmt=(
        select(Report).where(Report.status==ReportStatus.PENDING_REVIEW).order_by(Report.created_at.asc()).limit(limit)
    )
    result=await db.execute(stmt)
    reports=result.scalars().all()
    return {
        "count": len(reports),
        "reports": [ReportDetail.model_validate(r) for r in reports]
    }

@router.get("/dead-letter-queue")
async def get_dead_letter_queue(
    resolved: bool = Query(default=False),
    db: AsyncSession=Depends(get_db)
):
    stmt=(
        select(DeadLetterQueue)
        .where(DeadLetterQueue.resolved==resolved)
        .order_by(DeadLetterQueue.created_at.desc())
    )

    result=db.execute(stmt)
    entries=result.scalars().all()
    return {"count": len(entries), "entries":[
        {"id": str(e.id),
        "report_id": str(e.report_id),
        "phase": e.phase,
        "error_detail": e.error_detail,
        "retry_count": e.retry_count,
        "created_at": e.created_at.isoformat()}
        for e in entries
    ]}

@router.patch("/review-queue/{report_id}/approve")
async def approve_review(report_id: str, db: AsyncSession=Depends(get_db)):
    import uuid
    from app.db.models import LifecycleEvent
    from app.core.redis import get_redis_pool
    import redis.asyncio as aioredis
    from app.core.config import get_settings

    settings=get_settings()
    report = await db.get(Report, uuid.UUID(report_id))
    if not report: 
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Report not found")

    from_status=report.status
    report.status=ReportStatus.ANALYZED
    db.add(LifecycleEvent(
        report_id=report.id,
        from_status=from_status,
        to_status=ReportStatus.ANALYZED,
        detail="Manually approved by human reviewer"
    ))
    await db.commit()

    pool = get_redis_pool()
    redis_client = aioredis.Redis(connection_pool=pool)
    await redis_client.rpush(settings.knowledge_queue_key, str(report.id))
    await redis_client.aclose()
    return {"status":"approved", "report_id":report_id}