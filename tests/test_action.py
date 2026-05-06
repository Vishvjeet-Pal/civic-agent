import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.report import (
    ActionPlan, DetectedIssue, PerceptionResult, RecommendedTool
)

def make_plan(report_id=None) -> ActionPlan:
    return ActionPlan(
        report_id=report_id or uuid.uuid4(),
        issue_type="pothole",
        statue_ref="Municipal Code 14.2.3",
        severity="high",
        recommended_tools=[
            RecommendedTool.GEOCODE,
            RecommendedTool.SEND_REPORT,
            RecommendedTool.LOG_LEDGER,
        ], 
        context_summary="Large pothole detected. 14.2.3 mandates repair within 48 hours.",
        requires_human_review=False,
    )

def make_perception(report_id=None) -> PerceptionResult:
    return PerceptionResult(
        report_id=report_id or uuid.uuid4(),
        summary="Large pothole on main road.",
        confidence_score=0.92,
        issues=[
            DetectedIssue(
                type="pothole",
                bbox_ymin=550, bbox_xmin=210,
                bbox_ymax=670, bbox_xmax=390,
                severity=4,
                description="Hazardous pothole in motorcycle lane.",
            )
        ],
        gps_latitude=30.7046,
        gps_longitude=76.7179,
    )

@pytest.mark.asyncio 
async def test_reverse_geocode_called_with_correct_params():
    with patch("app.services.mcp_tools.httpx.AsyncClient") as mock_cls:
        mock_client=AsyncMock()
        mock_cls.return_value.__aenter__.return_value=mock_client
        mock_client.get.return_value=AsyncMock(
            status_code=200,
            json=lambda: {
                "display_name": "Main Street, Ludhiana, Punjab, India",
                "address": {"road": "Main Street", "city": "Ludhiana"}
                }
        )
        mock_client.get.return_value.raise_for_status = lambda: None

        from app.services.mcp_tools import reverse_geocode
        result = await reverse_geocode(latitude=30.7046, longitude=76.7179)

        assert "address" in result
        assert "Ludhiana" in result["address"]

@pytest.mark.asyncio 
async def test_tool_retry_on_failure():
    call_count=0

    async def flaky_call(*args, **kwargs):
        nonlocal call_count
        call_count+=1
        if call_count<3:
            raise ConnectionError("timeout")
        return MagicMock(
            status_code=200,
            json=lambda: {"display_name": "Some Road", "address": {}},
            raise_for_status=lambda: None,
        )
    with patch("app.services.mcp_tools.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_client.get = flaky_call
        from app.services.mcp_tools import reverse_geocode
        result = await reverse_geocode(latitude=30.7046, longitude=76.7179)

    assert call_count == 3
    assert result["address"] == "Some Road"


def test_tool_schemas_cover_all_tools():
    from app.services.mcp_tools import TOOL_SCHEMAS
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "reverse_geocode" in names
    assert "send_civic_report" in names
    assert "log_to_official_ledger" in names


@pytest.mark.asyncio
async def test_run_action_routes_to_dlq_on_llama_failure():
    from app.services.action import run_action
    import uuid as _uuid

    report_id = _uuid.uuid4()
    mock_report = MagicMock()
    mock_report.id = report_id
    mock_report.status = MagicMock(value="ANALYZED")
    mock_report.provided_address = None
    mock_report.action_plan = make_plan(report_id).model_dump(mode="json")
    mock_report.perception_result = make_perception(report_id).model_dump(mode="json")

    mock_db = AsyncMock()
    mock_db.get.return_value = mock_report
    mock_redis = AsyncMock()

    with patch("app.services.action.get_tool_calls", side_effect=RuntimeError("Groq down")):
        result = await run_action(report_id, mock_db, mock_redis)

    assert result is False
    mock_db.add.assert_called()