"""
Deduplication service: checks whether a civic issue has already been
reported recently for the same location and issue type.

Updated to support multi-user subscriptions to the same report.
"""
from datetime import datetime, timedelta, timezone
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Report, ReportSubscriber, User

logger = get_logger(__name__)

async def is_duplicate(
    db: AsyncSession, 
    issue_type: str, 
    lat: float | None, 
    lon: float | None,
    user: User | None = None
) -> tuple[bool, str | None]:
    """
    Checks if a similar issue exists within 100 meters and was created in the last 24 hours.
    If a user is provided and it's a duplicate, it adds them as a subscriber.
    """
    if lat is None or lon is None:
        return False, None

    settings = get_settings()
    threshold_time = datetime.now(timezone.utc) - timedelta(hours=24)

    # Simple bounding box check (~100m)
    lat_delta = 0.001
    lon_delta = 0.001

    stmt = select(Report).where(
        and_(
            Report.gps_latitude.between(lat - lat_delta, lat + lat_delta),
            Report.gps_longitude.between(lon - lon_delta, lon + lon_delta),
            Report.created_at >= threshold_time
        )
    )
    
    result = await db.execute(stmt)
    reports = result.scalars().all()

    for report in reports:
        # Check if the perception result contains the same issue type
        # perception_result is a JSONB field
        p_res = report.perception_result
        if p_res and "issues" in p_res:
            existing_types = [i["type"] for i in p_res["issues"]]
            if issue_type in existing_types:
                # If a user is logged in, link them to this report
                if user:
                    # Check if already subscribed
                    sub_check = await db.execute(
                        select(ReportSubscriber).where(
                            and_(
                                ReportSubscriber.report_id == report.id,
                                ReportSubscriber.user_id == user.id
                            )
                        )
                    )
                    if not sub_check.scalar_one_or_none():
                        db.add(ReportSubscriber(report_id=report.id, user_id=user.id))
                        # Note: We don't commit here, the caller handles it
                
                return True, str(report.id)

    return False, None
