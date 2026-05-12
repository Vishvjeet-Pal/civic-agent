import enum, uuid
from datetime import datetime
from sqlalchemy import Enum, Float, Index, String, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class UserRole(str, enum.Enum):
    CITIZEN = "CITIZEN"
    AUTHORITY = "AUTHORITY"

class AuthorityStatus(str, enum.Enum):
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"

class ReportStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    ANALYZED = "ANALYZED"
    ACTIONED = "ACTIONED"
    FAILED = "FAILED"
    PENDING_REVIEW = "PENDING_REVIEW"

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.CITIZEN)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Report(Base):
    __tablename__='reports'

    id: Mapped[uuid.UUID]=mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[ReportStatus]=mapped_column(
        Enum(ReportStatus), nullable=False, default=ReportStatus.RECEIVED
    )
    # Authority tracking
    authority_status: Mapped[AuthorityStatus] = mapped_column(
        Enum(AuthorityStatus), default=AuthorityStatus.PENDING
    )
    assigned_department_emails: Mapped[list[str] | None] = mapped_column(JSONB, default=[])
    reporter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
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
    action_result: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    lifecycle_events: Mapped[list["LifecycleEvent"]] = relationship(
        back_populates="report", cascade="all, delete-orphan", passive_deletes=True
    )
    subscribers: Mapped[list["ReportSubscriber"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    
    @property
    def image_url(self) -> str | None:
        if self.image_path:
            return f"/api/v1/reports/{self.id}/image"
        return None

    __table_args__ = (
        Index("ix_reports_status", "status"),
        Index("ix_reports_created_at", "created_at"),
        Index("ix_reports_depts", "assigned_department_emails", postgresql_using="gin"),
    )

class ReportSubscriber(Base):
    __tablename__ = "report_subscribers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    report: Mapped["Report"] = relationship(back_populates="subscribers")

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