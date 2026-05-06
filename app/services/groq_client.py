"""Calls Llama 3.3 70B to decide which MCP tools to invoke based on the validated ActionPlan."""

import json
from typing import Any
import httpx
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.report import ActionPlan, PerceptionResult
from app.schemas.tools import TOOL_SCHEMAS

logger=get_logger(__name__)

_GROQ_BASE = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = """You are an action execution agent for a civic reporting system.

You receive a validated ActionPlan JSON and must call the appropriate tools
in the correct order to file the civic report:

1. Always call send_civic_report to notify the municipal department.
2. Always call log_to_official_ledger last to close the record.

Do not call tools not listed in recommended_tools.
Call tools one at a time — never in parallel."""

def _build_user_message(plan: ActionPlan, perception: PerceptionResult) -> str:
    gps = (
        f"latitude={perception.gps_latitude}, longitude={perception.gps_longitude}"
        if perception.gps_latitude is not None
        else "GPS not available"
    )
    return f"""ACTION PLAN:
{plan.model_dump_json(indent=2)}

PERCEPTION SUMMARY:
- Scene: {perception.summary}
- Issues: {len(perception.issues)} detected
- Confidence: {perception.confidence_score:.2f}
- Location: {gps}

Execute the recommended_tools in order now."""

async def get_tool_calls(plan: ActionPlan, perception: PerceptionResult) -> list[dict[str, Any]]:
    """
    Ask Llama 3.3 70B which tools to call and in what order.
    Returns a list of tool call dicts: [{name, arguments}].
    """
    settings = get_settings()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role":"user", "content":_build_user_message(plan, perception)},
    ]

    # Filter tool schemas to only those in the plan (excluding reverse_geocode which is handled automatically)
    allowed = {t.value for t in plan.recommended_tools if t.value != "reverse_geocode"}
    active_schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]

    collected_calls: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=settings.groq_timeout_seconds) as client:
        # Agentic loop: Llama calls one tool at a time until done
        for _ in range(10):  # hard cap — prevents infinite loops
            response = await client.post(
                f"{_GROQ_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.groq_model,
                    "messages": messages,
                    "tools": active_schemas,
                    "tool_choice": "auto",
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice["finish_reason"]

            # Append assistant turn to history
            messages.append(message)

            if finish_reason == "tool_calls":
                for tc in message.get("tool_calls", []):
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    collected_calls.append({"name": name, "arguments": args, "id": tc["id"]})
                    logger.info("llama_tool_call", tool=name, args=args)

                    # Append a stub tool result so the model can continue
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"status": "executing"}),
                    })

            elif finish_reason == "stop":
                # Model is done calling tools
                break

    logger.info(
        "llama_tool_selection_complete",
        report_id=str(plan.report_id),
        tools_selected=[c["name"] for c in collected_calls],
    )
    return collected_calls