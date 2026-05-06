"""
Gemini client using the official google-genai SDK.
Takes the PerceptionResult + RAG context chunks and returns a validated ActionPlan.
Retries once with a stricter prompt on validation failure, then routes to human review.
"""

import asyncio
import json
from pathlib import Path
from google import genai
from google.genai import types
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.report import ActionPlan, PerceptionResult, RecommendedTool

logger = get_logger(__name__)

def _get_mail_categories() -> str:
    """Load issue categories from mails.json for prompt injection."""
    mails_path = Path("municipal_docs/mails.json")
    if not mails_path.exists():
        return "fallen_tree_blockage, illegal_dumping"
    try:
        with open(mails_path, "r") as f:
            data = json.load(f)
        categories = set()
        for city_data in data.values():
            categories.update(city_data.get("department_routing", {}).keys())
        return ", ".join(sorted(categories))
    except Exception:
        return "fallen_tree_blockage, illegal_dumping"

SYSTEM_PROMPT_TEMPLATE = """You are a legal compliance engine for a municipal civic reporting system.
You will receive:
1. A list of detected civic issues with severity scores (1-5).
2. Relevant excerpts from the municipal code.

Your job is to produce a legally-grounded action plan.

YOU MUST respond ONLY with a JSON object. No markdown, no preamble.
Schema:
{{
  "report_id": "UUID string",
  "issue_type": "string (MUST BE ONE OF: {categories})",
  "statute_ref": "string or null",
  "severity": "low | medium | high | critical | null",
  "recommended_tools": ["send_civic_report" | "log_to_official_ledger" | "reverse_geocode"],
  "context_summary": "string",
  "requires_human_review": boolean
}}

Severity mapping from detected issue scores:
    1-2 → low, 3 → medium, 4 → high, 5 → critical. Set to null if no clear issues.
Always include all three recommended_tools unless there is no GPS data
(omit reverse_geocode if GPS coordinates are null).
If no action is needed, recommended_tools can be an empty list [].
Always set requires_human_review to false.
"""
# Set requires_human_review to true if the statute is ambiguous or no law clearly applies.


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
    CONFIDENCE SCORE: {perception.confidence_score: .2f}
    {gps_text}

    DETECTED ISSUES:
    {issues_text}

    MUNICIPAL CODE CONTEXT:
    {context_text}

    Produce the JSON action plan now."""

async def _call_gemini_genai(prompt: str, system: str) -> str:
    """Modern Google GenAI SDK call."""
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    
    # Use the async generate method via .aio
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=2048,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
        )
    )
    
    if not response.text:
        # Check if it was blocked or had another finish reason
        reason = response.candidates[0].finish_reason if response.candidates else "unknown"
        raise ValueError(f"Empty response from Gemini. Finish reason: {reason}")
        
    return response.text

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
    categories = _get_mail_categories()
    base_system = SYSTEM_PROMPT_TEMPLATE.format(categories=categories)

    for attempt in range(1, 3):
        strict_system = base_system if attempt == 1 else (
            base_system + "\n\nCRITICAL: Your previous response failed JSON validation. "
            "Return ONLY the raw JSON object. Absolutely no extra text."
        )
        raw = "<no response>"
        try:
            raw = await _call_gemini_genai(user_prompt, strict_system)
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
                raw_response=raw
            )
            if attempt == 2:
                raise ValueError(
                    f"Gemini ActionPlan validation failed after 2 attempts: {exc}"
                ) from exc
            await asyncio.sleep(1)