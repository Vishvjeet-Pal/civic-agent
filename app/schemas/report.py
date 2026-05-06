import uuid
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices

class IncomingReport(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(..., pattern=r"^image/(jpeg|png|webp)$")
    address: str | None = None

class BoundingBox(BaseModel):
    ymin: int = Field(..., ge=0, le=1000)
    xmin: int = Field(..., ge=0, le=1000)
    ymax: int = Field(..., ge=0, le=1000)
    xmax: int = Field(..., ge=0, le=1000)


class DetectedIssue(BaseModel):
    type: str
    bbox_ymin: int
    bbox_xmin: int
    bbox_ymax: int
    bbox_xmax: int
    severity: int = Field(..., ge=1, le=5)
    description: str


class PerceptionResult(BaseModel):
    report_id: uuid.UUID
    summary: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    issues: list[DetectedIssue] = Field(default_factory=list)
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    captured_at: datetime | None = None
    provided_address: str | None = None
    low_confidence: bool = False
    issue_count: int = 0
    issue_label: str = "none"

    @model_validator(mode="after")
    def derive_fields(self) -> "PerceptionResult":
        self.issue_count = len(self.issues)
        if self.issues:
            self.issue_label = self.issues[0].type
        return self

    @field_validator("low_confidence", mode="before")
    @classmethod
    def derive_low_confidence(cls, v, info):
        return v

class RecommendedTool(str, Enum):
    SEND_REPORT = "send_civic_report"
    LOG_LEDGER = "log_to_official_ledger"
    GEOCODE = "reverse_geocode"


class ActionPlan(BaseModel):
    report_id: uuid.UUID
    issue_type: str = Field(..., min_length=1)
    statute_ref: str | None = None          # e.g. "Municipal Code §14.2.3"
    severity: str | None = Field(default=None, pattern=r"^(low|medium|high|critical)$")
    recommended_tools: list[RecommendedTool] = Field(default_factory=list)
    context_summary: str = Field(..., min_length=10, max_length=1000)
    requires_human_review: bool = False

class ReportResponse(BaseModel):
    id: uuid.UUID
    status: str
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportDetail(BaseModel):
    id: uuid.UUID
    status: str
    original_filename: str
    gps_latitude: float | None
    gps_longitude: float | None
    captured_at: datetime | None
    provided_address: str | None
    confidence_score: float | None
    perception_result: dict[str, Any] | None
    action_plan: dict[str, Any] | None
    action_result: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}