"""
Action service: loads ActionPlan + PerceptionResult, calls Groq/Llama
for tool selection, executes MCP tools in order, handles retries and DLQ.
"""
import uuid
import re
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import DeadLetterQueue, LifecycleEvent, Report, ReportStatus
from app.schemas.report import ActionPlan, PerceptionResult
from app.services.groq_client import get_tool_calls
from app.services.mcp_tools import (
    log_to_official_ledger,
    reverse_geocode,
    send_civic_report,
)

logger = get_logger(__name__)

async def _transition(db: AsyncSession, report: Report, to: ReportStatus, detail: str) -> None:
    report.status = to
    db.add(LifecycleEvent(
        report_id=report.id,
        from_status=report.status,
        to_status=to, 
        detail=detail, 
    ))
    await db.flush()

def _parse_coordinates(address: str) -> tuple[float, float] | None:
    if not address:
        return None
    # Matches "lat, lon" or "lat,lon"
    pattern = r'^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$'
    if not re.match(pattern, address.strip()):
        return None
    try:
        lat_s, lon_s = address.split(',')
        return float(lat_s.strip()), float(lon_s.strip())
    except Exception:
        return None

async def run_action(report_id: uuid.UUID, db: AsyncSession, redis: aioredis.Redis,) -> bool:
    """
    Full action pipeline. Returns True on success, False on DLQ.
    """
    report = await db.get(Report, report_id)
    if not report:
        logger.error("action_report_not_found", report_id=str(report_id))
        return False

    if not report.action_plan or not report.perception_result:
        logger.error("action_missing_contracts", report_id=str(report_id))
        await _transition(db, report, ReportStatus.FAILED, "Missing ActionPlan or PerceptionResult")
        await db.commit()
        return False

    plan = ActionPlan.model_validate(report.action_plan)
    perception = PerceptionResult.model_validate(report.perception_result)

    try:
        tool_calls= await get_tool_calls(plan, perception)
    except Exception as exc:
        logger.error("action_llama_failed", report_id=str(report_id), error=str(exc))
        db.add(DeadLetterQueue(
            report_id=report_id,
            phase="action_llama",
            error_detail=str(exc),
            retry_count=3,
        ))
        await _transition(db, report, ReportStatus.FAILED, f"Llama tool selection failed: {exc}")
        await db.commit()
        return False
    resolved_address: str | None = report.provided_address
    provided_coords = _parse_coordinates(resolved_address) if resolved_address else None
    tool_results: dict[str, object] = {}

    # ── Automatic Geocoding ────────────────────────────────────────────────
    # We resolve coordinates to addresses automatically before calling the AI
    lat, lon = None, None
    if provided_coords:
        lat, lon = provided_coords
        logger.info("geocoding_provided_coordinates", report_id=str(report_id), coords=provided_coords)
    elif not resolved_address and perception.gps_latitude is not None and perception.gps_longitude is not None:
        lat, lon = perception.gps_latitude, perception.gps_longitude
        logger.info("geocoding_perception_gps", report_id=str(report_id), lat=lat, lon=lon)

    if lat is not None and lon is not None:
        try:
            result = await reverse_geocode(latitude=lat, longitude=lon)
            resolved_address = result.get("address")
            tool_results["reverse_geocode"] = result
            logger.info("automatic_geocode_complete", report_id=str(report_id), address=resolved_address)
        except Exception as exc:
            logger.error("automatic_geocode_failed", report_id=str(report_id), error=str(exc))

    for call in tool_calls:
        name = call["name"]
        try:
            if name == "send_civic_report":
                result = await send_civic_report(
                    plan=plan,
                    perception=perception,
                    address=resolved_address
                )
                tool_results["send_civic_report"]  = result

            elif name == "log_to_official_ledger":
                result = await log_to_official_ledger(
                    report_id=report_id,
                    plan=plan,
                    address=resolved_address,
                    db=db,
                )
                tool_results["log_to_official_ledger"] = result
            logger.info("tool_executed", tool=name, report_id=str(report_id))

        except RuntimeError as exc:
            logger.error("tool_exhausted", tool=name, report_id=str(report_id), error=str(exc))
            db.add(DeadLetterQueue(
                report_id=report_id,
                phase=f"action_{name}",
                error_detail=str(exc),
                retry_count=3,
            ))
            await _transition(db, report, ReportStatus.FAILED, f"Tool '{name}' failed after retries: {exc}")
            await db.commit()
            return False
    report.action_result = {
        "tools_executed": list(tool_results.keys()),
        "resolved_address": resolved_address,
        **tool_results,
    }
    await _transition(
        db, report, ReportStatus.ACTIONED,
        f"All tools executed successfully: {list(tool_results.keys())}"
    )
    await db.commit()

    logger.info(
        "action_complete",
        report_id=str(report_id),
        tools=list(tool_results.keys()),
        address=resolved_address,
    )
    return True