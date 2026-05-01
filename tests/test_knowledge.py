import json 
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.report import ActionPlan, PerceptionResult, DetectedIssue, RecommendedTool

def make_perception(issues=None) -> PerceptionResult:
    return PerceptionResult(
        report_id=uuid.uuid4(),
        summary="Pothole on main road.",
        overall_confidence=0.92,
        issues=issues or [
            DetectedIssue(
                type="pothole",
                bbox_ymin=550, bbox_xmin=210,
                bbox_ymax=670, bbox_xmax=390,
                severity=4,
                description="Large pothole, hazard to motorcyclists."
            )
        ],
        gps_latitude=30.7046,
        gps_longitude=76.7179
    )

def test_cache_key_is_deterministic():
    from app.services.knowledge import _cache_key
    key1=_cache_key(["pothole","graffiti"])
    key2=_cache_key(["graffiti", "pothole"])
    key3=_cache_key(["pothole"])

    assert key1==key2
    assert key1 != key3
    assert key1.startswith("rag:")

def test_action_plan_schema_validation():
    plan = ActionPlan(
        report_id=uuid.uuid4(),
        issue_type="pothole",
        statute_ref="Municipal Code §14.2.3",
        severity="high",
        recommended_tools=[
            RecommendedTool.SEND_REPORT,
            RecommendedTool.LOG_LEDGER,
            RecommendedTool.GEOCODE
        ],
        context_summary="A large pothole was detected. §14.2.3 mandates repair within 48h.",
        requires_human_review=False
    )
    assert plan.severity == "high"
    assert len(plan.recommeded_tools)==3

def test_action_plan_rejects_invalid_severity():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ActionPlan(
            report_id=uuid.uuid4(),
            issue_type="pothole",
            severity="extreme",  
            recommended_tools=[RecommendedTool.SEND_REPORT],
            context_summary="Test.",
        )

@pytest.mark.asyncio 
async def test_run_knowledge_cache_hit():
    from app.services.knowledge import run_knowledge

    report_id = uuid.uuid4()
    cached_plan = {
        "issue_type": "pothole",
        "statute_ref": "§14.2.3",
        "severity": "high",
        "recommended_tools": ["send_civic_report", "log_to_official_ledger", "reverse_geocode"],
        "context_summary": "Pothole requires repair per §14.2.3.",
        "requires_human_review": False,
    }
    mock_report=MagicMock()
    mock_report.id = report_id
    mock_report.status.value = "ANALYZED"
    mock_report.perception_result = make_perception().model_dump(mode="json")
    mock_db=AsyncMock()
    mock_db.get.return_value = mock_report
    mock_redis=AsyncMock()
    mock_redis.get.return_value = json.dumps(cached_plan)

    with patch("app.services.knowledge.get_settings") as mock_settings:
        mock_settings.return_value.action_queue_key="action:queue"
        mock_settings.return_value.rag_cache_ttl_seconds=604800
        result=await run_knowledge(report_id, mock_db, mock_redis)
    
    assert result is not None
    assert result.issue_type =="pothole"
    mock_redis.rpush.assert_called_once_with("action:queue", str(report_id))