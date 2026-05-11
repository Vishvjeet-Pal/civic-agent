"""
Authority router: handles report resolution and department-specific actions.
"""
import uuid
import smtplib
import asyncio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.db.models import Report, ReportSubscriber, AuthorityStatus, User, UserRole, ReportStatus, LifecycleEvent
from app.services.auth import get_current_authority

logger = get_logger(__name__)
router = APIRouter(prefix="/authority", tags=["authority"])

async def send_resolution_email(email: str, report_id: uuid.UUID, issue_type: str, address: str | None):
    """Sends a notification email to the citizen."""
    settings = get_settings()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Resolved] Update on your report: {issue_type}"
    msg["From"] = settings.smtp_user
    msg["To"] = email

    body = f"""
    <html><body>
    <h2>Action Taken: Report Resolved</h2>
    <p>Good news! The {issue_type} you reported has been marked as <strong>Resolved</strong> by the department.</p>
    <p><strong>Report ID:</strong> {report_id}</p>
    <p><strong>Location:</strong> {address or "Provided Location"}</p>
    <p>Thank you for using Civic Guardian to help improve your city.</p>
    <hr>
    <p style="color:#999;font-size:12px">This is an automated notification.</p>
    </body></html>
    """
    msg.attach(MIMEText(body, "html"))

    def _send():
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send)

@router.post("/resolve/{report_id}")
async def resolve_report(
    report_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    authority: User = Depends(get_current_authority)
):
    """Mark a report as resolved and notify all subscribers."""
    # 1. Fetch report and subscribers
    stmt = select(Report).where(Report.id == report_id).options(selectinload(Report.subscribers))
    result = await db.execute(stmt)
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    if not report.assigned_department_emails or authority.email not in report.assigned_department_emails:
        raise HTTPException(status_code=403, detail="Report not assigned to your department")

    if report.authority_status == AuthorityStatus.RESOLVED:
        return {"message": "Already resolved"}

    # 2. Update status
    report.authority_status = AuthorityStatus.RESOLVED
    db.add(LifecycleEvent(
        report_id=report.id,
        from_status=report.status,
        to_status=report.status, # Keep pipeline status same, or maybe set to something final
        detail=f"Authority ({authority.full_name}) marked as RESOLVED"
    ))

    # 3. Get all subscriber emails
    sub_stmt = select(User.email).join(ReportSubscriber, User.id == ReportSubscriber.user_id).where(
        ReportSubscriber.report_id == report_id
    )
    sub_result = await db.execute(sub_stmt)
    emails = sub_result.scalars().all()

    # 4. Notify everyone
    issue_type = report.perception_result.get("issues", [{}])[0].get("type", "Issue") if report.perception_result else "Issue"
    
    tasks = [send_resolution_email(email, report_id, issue_type, report.provided_address) for email in emails]
    if tasks:
        # Fire and forget or wait? Let's wait for now to be sure.
        await asyncio.gather(*tasks, return_exceptions=True)

    await db.commit()
    logger.info("report_resolved_by_authority", report_id=str(report_id), dept=authority.email)
    
    return {"status": "success", "notified_count": len(emails)}


@router.post("/resolve-bulk")
async def resolve_bulk(
    issue_type: str,
    address: str,
    db: AsyncSession = Depends(get_db),
    authority: User = Depends(get_current_authority)
):
    """Bulk resolve all reports with the same issue type and address."""
    # We query for reports assigned to this dept that are PENDING.
    # We'll filter the issue_type and address in Python to be more flexible (case-insensitive, action_result check).
    
    stmt = select(Report).where(
        and_(
            Report.assigned_department_emails.contains([authority.email]),
            Report.authority_status == AuthorityStatus.PENDING
        )
    ).options(selectinload(Report.subscribers))
    
    result = await db.execute(stmt)
    reports = result.scalars().all()
    
    count = 0
    total_notified = 0
    
    # Normalize inputs for comparison
    target_issue = issue_type.strip().lower()
    target_address = address.strip().lower()
    
    for report in reports:
        # 1. Check Address Match
        # Match against either the raw provided_address OR the resolved_address in action_result
        db_address = str(report.provided_address or "").strip().lower()
        resolved_addr = str((report.action_result or {}).get("resolved_address") or "").strip().lower()
        
        if target_address != db_address and target_address != resolved_addr:
            continue
            
        # 2. Check Issue Type Match
        issues = (report.perception_result or {}).get("issues", [])
        if not isinstance(issues, list):
            issues = [issues]
            
        match_found = False
        for issue in issues:
            if issue.get("type", "").strip().lower() == target_issue:
                match_found = True
                break
        
        if match_found:
            # Resolve it
            report.authority_status = AuthorityStatus.RESOLVED
            
            # 3. Notify subscribers
            sub_stmt = select(User.email).join(ReportSubscriber, User.id == ReportSubscriber.user_id).where(
                ReportSubscriber.report_id == report.id
            )
            e_res = await db.execute(sub_stmt)
            emails = e_res.scalars().all()
            
            tasks = [send_resolution_email(email, report.id, issue_type, report.provided_address) for email in emails]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            count += 1
            total_notified += len(emails)

    await db.commit()
    return {"status": "success", "resolved_count": count, "total_notified": total_notified}
