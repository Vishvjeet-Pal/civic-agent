"""
Deduplication service: checks whether a civic issue has already been
reported recently for the same location and issue type.

Called by the live-stream capture endpoint before submitting a full report.
"""
import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Report, ReportStatus

logger = get_logger(__name__)

# How close two GPS points must be (in decimal degrees) to be "same location"
# ~0.0005° ≈ 55 metres
LOCATION_TOLERANCE = 0.0005

# How many hours before the same issue+location can be re-reported
DEDUP_WINDOW_HOURS = 24


async def is_duplicate(
    db: AsyncSession,
    issue_type: str,
    latitude: float | None,
    longitude: float | None,
) -> tuple[bool, str | None]:
    """
    Returns (is_duplicate, existing_report_id).
    Considers a report a duplicate if:
      - Same issue_type (from action_plan->issue_type)
      - GPS within LOCATION_TOLERANCE degrees
      - Created within DEDUP_WINDOW_HOURS
      - Status not FAILED (i.e. it was actually processed)
    If no GPS available, deduplication is skipped (returns False).
    """
    if latitude is None or longitude is None:
        return False, None

    cutoff = datetime.now(timezone.utc) - timedelta(hours=DEDUP_WINDOW_HOURS)

    stmt = select(Report).where(
        and_(
            Report.created_at >= cutoff,
            Report.status.in_([
                ReportStatus.RECEIVED,
                ReportStatus.PROCESSING,
                ReportStatus.ANALYZED,
                ReportStatus.ACTIONED,
                ReportStatus.PENDING_REVIEW,
            ]),
            Report.gps_latitude.between(
                latitude - LOCATION_TOLERANCE,
                latitude + LOCATION_TOLERANCE,
            ),
            Report.gps_longitude.between(
                longitude - LOCATION_TOLERANCE,
                longitude + LOCATION_TOLERANCE,
            ),
        )
    )
    result = await db.execute(stmt)
    candidates = result.scalars().all()

    for report in candidates:
        plan = report.action_plan or {}
        existing_type = plan.get("issue_type", "").lower()
        if existing_type and (issue_type.lower() in existing_type or existing_type in issue_type.lower()):
            logger.info(
                "dedup_hit",
                existing_report=str(report.id),
                issue_type=issue_type,
                lat=latitude,
                lon=longitude,
            )
            return True, str(report.id)

    return False, None
