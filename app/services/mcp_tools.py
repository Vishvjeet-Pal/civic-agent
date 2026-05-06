"""
MCP tools for Civic Agent.
Each tool is an async function with its own retry wrapper.
"""

import asyncio
import json
from pathlib import Path
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any 
import httpx
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.report import ActionPlan, PerceptionResult

logger = get_logger(__name__)

def _with_retry(max_attempts: int = 3):
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_err = exc
                    wait = 2 ** (attempt-1)
                    logger.warning(
                        "tool_retry",
                        tool=fn.__name__,
                        attempt=attempt,
                        error=str(exc),
                        retry_in=wait 
                    )
                    if attempt<max_attempts:
                        await asyncio.sleep(wait)
            raise RuntimeError(
                f"Tool '{fn.__name__}' failed after {max_attempts} attempts: {last_err}"
            )
        wrapper.__name__=fn.__name__
        return wrapper
    return decorator

@_with_retry(max_attempts=3)
async def reverse_geocode(latitude: float, longitude: float) -> dict[str, Any]:
    """Convert GPS coordinates to a human-readable street address."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            settings.geocoding_api_url,
            params={"lat": latitude, "lon": longitude, "format": "json"},
            headers={"User-Agent": settings.geocoding_user_agent},
        )

        response.raise_for_status()
        data=response.json()

    address = data.get("display_name", f"{latitude}, {longitude}")
    logger.info("geocode_complete", address=address[:60])
    return{"address": address, "raw": data.get("address", {})}

def _get_department_email(issue_type: str) -> str:
    """
    Robustly look up the primary contact for a given issue type in mails.json.
    Tries exact match, then substring match on keys, then search in descriptions.
    """
    settings = get_settings()
    default_email = settings.municipal_email
    
    mails_path = Path("municipal_docs/mails.json")
    if not mails_path.exists():
        logger.warning("mails_json_not_found", path=str(mails_path))
        return default_email
        
    try:
        with open(mails_path, "r") as f:
            data = json.load(f)
            
        # Normalize input
        query = issue_type.lower().replace("_", " ").replace("-", " ").strip()
        query_words = set(query.split())
        stop_words = {"and", "or", "the", "to", "of", "in", "with", "due", "at"}
        query_keywords = query_words - stop_words
        
        candidates = [] # List of (score, email)
        
        for city_data in data.values():
            routing = city_data.get("department_routing", {})
            
            for key, info in routing.items():
                norm_key = key.lower().replace("_", " ").replace("-", " ")
                description = info.get("issue_description", "").lower().replace("/", " ").replace("-", " ")
                contact = info.get("primary_contact")
                
                if not contact:
                    continue
                    
                # 1. Exact match (Score 0)
                if query == norm_key:
                    logger.info("exact_match_found", issue_type=issue_type, email=contact)
                    return contact
                
                # 2. Key substring match (Score 10)
                if query in norm_key or norm_key in query:
                    candidates.append((10, contact))
                    continue

                # 3. Word overlap with key (Score 20 - matches)
                key_words = set(norm_key.split()) - stop_words
                intersection_key = query_keywords.intersection(key_words)
                if intersection_key:
                    # Better score for more word matches
                    score = 20 - len(intersection_key)
                    candidates.append((score, contact))
                    
                # 4. Word overlap with description (Score 30 - matches)
                desc_words = set(description.split()) - stop_words
                intersection_desc = query_keywords.intersection(desc_words)
                if intersection_desc:
                    score = 30 - len(intersection_desc)
                    candidates.append((score, contact))
        
        if candidates:
            # Sort by priority score (lower is better)
            candidates.sort(key=lambda x: x[0])
            best_email = candidates[0][1]
            logger.info("fuzzy_match_found", issue_type=issue_type, email=best_email, best_score=candidates[0][0])
            return best_email

        logger.info("department_email_not_found", issue_type=issue_type)
        return default_email
    except Exception as exc:
        logger.error("mails_json_parse_failed", error=str(exc))
        return default_email

def _build_email_body(plan: ActionPlan, perception: PerceptionResult, address: str | None) -> str:
    issues_html = "".join(
        f"<li><strong>{i.type}</strong> (severity {i.severity}/5): {i.description}</li>"
        for i in perception.issues
    )
    location=address or (
        f"{perception.gps_latitude:.5f}, {perception.gps_longitude:.5f}"
        if perception.gps_latitude else "Not available"
    )
    return f"""
    <html><body style="font-family:sans-serif;max-width:600px">
    <h2 style="color:#c0392b">Civic Issue Report — {plan.issue_type.title()}</h2>
    <table style="border-collapse:collapse;width:100%">
      <tr><td style="padding:6px;color:#666">Report ID</td>
          <td style="padding:6px"><code>{plan.report_id}</code></td></tr>
      <tr style="background:#f8f8f8">
          <td style="padding:6px;color:#666">Location</td>
          <td style="padding:6px">{location}</td></tr>
      <tr><td style="padding:6px;color:#666">Severity</td>
          <td style="padding:6px"><strong>{(plan.severity or "N/A").upper()}</strong></td></tr>
      <tr style="background:#f8f8f8">
          <td style="padding:6px;color:#666">Statute</td>
          <td style="padding:6px">{plan.statute_ref or "N/A"}</td></tr>
    </table>
    <h3>Detected Issues</h3>
    <ul>{issues_html}</ul>
    <h3>Legal Summary</h3>
    <p>{plan.context_summary}</p>
    <hr>
    <p style="color:#999;font-size:12px">Generated by CivicAgent · Report ID: {plan.report_id}</p>
    </body></html>
    """

@_with_retry(max_attempts=3)
async def send_civic_report(plan: ActionPlan, perception: PerceptionResult, address: str | None=None,) -> dict[str, Any]:
    """Send a formatted civic report email to the municipal department."""

    settings = get_settings()

    msg = MIMEMultipart("alternative")
    severity_str = plan.severity.upper() if plan.severity else "UNKNOWN"
    msg["Subject"] = f"[CivicAgent] {severity_str} - {plan.issue_type.title()} at {address or 'Unknown Location'}"
    msg["From"] = settings.smtp_user
    
    # Dynamic routing
    to_email = _get_department_email(plan.issue_type)
    msg["To"] = to_email
    msg["X-Report_ID"] = str(plan.report_id)

    msg.attach(MIMEText(_build_email_body(plan, perception, address), "html"))

    loop = asyncio.get_running_loop()

    def _send():
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)

    await loop.run_in_executor(None, _send)

    logger.info(
        "civic_report_sent",
        report_id=str(plan.report_id),
        to=to_email,
        severity=plan.severity,
    )
    return {"sent":True, "to":to_email}

@_with_retry(max_attempts=3)
async def log_to_official_ledger(report_id: uuid.UUID, plan: ActionPlan, address: str | None, db,) -> dict[str, Any]:
    """Write the final action record to the PostgreSQL ledger."""
    from app.db.models import Report
    from sqlalchemy import update

    ledger_entry = {
        "resolved_address": address,
        "tools_called": [t.value for t in plan.recommended_tools],
    }

    stmt=(
        update(Report).where(Report.id==report_id).values(action_result=ledger_entry)
    )
    await db.execute(stmt)

    logger.info(
        "ledger_written",
        report_id=str(report_id),
        issue_type=plan.issue_type,
        severity=plan.severity
    )
    return ledger_entry 
    