"""
Gemini 1.5 Flash client.
Takes the PerceptionResult + RAG context chunks and returns a validated ActionPlan.
Retries once with a stricter prompt on validation failure, then routes to human review.
"""

import asyncio
import json
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.report import ActionPlan, PerceptionResult, RecommendedTool

logger = get_logger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
SYSTEM_PROMPT = """You are a legal compliance engine for a municipal civic reporting system.
You will receive:
1. A list of detected civic issues with severity scores (1-5).
2. Relevant excerpts from the municipal code.

Your job is to produce a legally-grounded action plan.

YOU MUST respond ONLY with valid JSON matching this exact schema - no markdown, no preamble:
{{
  "report_id": "<uuid from input>",
  "issue_type": "<primary issue type, e.g. 'pothole'>",
  "statute_ref": "<most relevant municipal code reference, e.g. 'Municipal Code §14.2.3', or null>",
  "severity": "<one of: low | medium | high | critical>",
  "recommended_tools": ["send_civic_report", "log_to_official_ledger", "reverse_geocode"],
  "context_summary": "<2-3 sentences: what the issue is, what the law says, what action is warranted>",
  "requires_human_review": false | true
}}

Severity mapping from detected issue scores:
    1-2 → low, 3 → medium, 4 → high, 5 → critical
Always include all three recommended_tools unless there is no GPS data
(omit reverse_geocode if GPS coordinates are null).
Set requires_human_review to true if the statute is ambiguous or no law clearly applies.
"""

def _build_user_prompt(perception: PerceptionResult, context_chunks: list[str]) -> str:
    issues_text = "\n".join(
        f" - {i.type} (severity {i.severity}/5): {i.description}"
        for i in perception.issues 
    ) or " - No issues detected"

    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No municipal code context available."

    gps_text = (
        f"GPS: {perception.gps_latitude}, {perception.gps_longitude}"
        if perception.gps_latitude is not None else "GPS: not available"
    )

    return f"""REPORT ID: {perception.report_id}
    SCENE SUMMARY: {perception.summary}
    OVERALL CONFIDENCE: {perception.overall_confidence: .2f}
    {gps_text}

    DETECTED ISSUES:
    {issues_text}

    MUNICIPAL CODE CONTEXT:
    {context_text}

    Produce the JSON action plan now."""

async def _call_gemini(prompt: str, system: str) -> str:
    """Raw Gemini API call. Returns the text content"""
    settings = get_settings()
    url = f"{_GEMINI_BASE}/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig":{
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data=response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError,IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {data}") from exc

def _parse_action_plan(raw_json: str) -> ActionPlan:
    clean = (
        raw_json.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    return ActionPlan.model_validate_json(clean)

async def build_action_plan(perception: PerceptionResult, context_chunks: list[str]) -> ActionPlan:
    """
    Call Gemini to produce a validated ActionPlan.
    On first validation failure, retries once with an explicit reminder.
    Raises ValueError after second failure — caller routes to human review.
    """

    user_prompt = _build_user_prompt(perception, context_chunks)

    for attempt in range(1,3):
        strict_system=SYSTEM_PROMPT if attempt == 1 else (
            SYSTEM_PROMPT+"\n\nCRITICAL: Your previous response failed JSON validation. "
            "Return ONLY the raw JSON object. Absolutely no extra text."
        )
        try:
            raw = await _call_gemini(user_prompt, strict_system)
            plan = _parse_action_plan(raw)
            logger.info(
                "gemini_action_plan_built",
                report_id=str(perception.report_id),
                issue_type=plan.issue_type,
                severity=plan.severity,
                statute=plan.statute_ref,
                attempt=attempt
            )
            return plan 
        except Exception as exc:
            logger.warning(
                "gemini_parse_failed",
                attempt=attempt,
                error=str(exc),
                raw_response=raw if "raw" in dir() else "<no response>"
            )
            if attempt == 2:
                raise ValueError(
                    f"Gemini ActionPlan validation failed after 2 attempts: {exc}"
                ) from exc
            await asyncio.sleep(1)