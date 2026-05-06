import enum, uuid
from datetime import datetime
from sqlalchemy import Enum, Float, Index, String, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class ReportStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    ANALYZED = "ANALYZED"
    ACTIONED = "ACTIONED"
    FAILED = "FAILED"
    PENDING_REVIEW = "PENDING_REVIEW"

class Report(Base):
    __tablename__='reports'

    id: Mapped[uuid.UUID]=mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[ReportStatus]=mapped_column(
        Enum(ReportStatus), nullable=False, default=ReportStatus.RECEIVED
    )
    original_filename: Mapped[str]=mapped_column(String(255), nullable=False)
    image_path: Mapped[str | None]=mapped_column(String(512))
    gps_latitude: Mapped[float | None]=mapped_column(Float)
    gps_longitude: Mapped[float | None]=mapped_column(Float)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provided_address: Mapped[str | None] = mapped_column(Text)

    perception_result: Mapped[dict | None] = mapped_column(JSONB)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    action_plan: Mapped[dict | None] = mapped_column(JSONB)

    # Action output (Phase 4)
    action_result: Mapped[dict | None] = mapped_column(JSONB)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Audit trail
    lifecycle_events: Mapped[list["LifecycleEvent"]] = relationship(
        back_populates="report", order_by="LifecycleEvent.created_at"
    )

    __table_args__ = (
        Index("ix_reports_status", "status"),
        Index("ix_reports_created_at", "created_at"),
    )


class LifecycleEvent(Base):
    __tablename__ = "lifecycle_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False
    )
    from_status: Mapped[ReportStatus | None] = mapped_column(Enum(ReportStatus))
    to_status: Mapped[ReportStatus] = mapped_column(Enum(ReportStatus), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    report: Mapped["Report"] = relationship(back_populates="lifecycle_events")

    __table_args__ = (Index("ix_lifecycle_report_id", "report_id"),)


class DeadLetterQueue(Base):
    __tablename__ = "dead_letter_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE")
    )
    phase: Mapped[str] = mapped_column(String(50), nullable=False)
    error_detail: Mapped[str] = mapped_column(Text, nullable=False)
    retry_count: Mapped[int] = mapped_column(default=0)
    resolved: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )