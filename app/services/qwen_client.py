"""
Vision client: calls LM Studio's OpenAI-compatible API to analyse civic images.

LM Studio runs on the host machine and exposes a local OpenAI-compatible server
at http://localhost:1234/v1.  From inside Docker it is reached via
host.docker.internal:1234.

Default model: qwen/qwen3-vl-4b  (load it in LM Studio first)
Override via LMSTUDIO_MODEL in .env.

The existing QwenResponse / Issue schema is preserved so no other code changes.
"""
import asyncio
import base64
import json
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError, AliasChoices

from app.core.config import get_settings
from app.core.logging import get_logger

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

SYSTEM_PROMPT_TEMPLATE = """You are an expert AI Municipal Inspector.
Analyze the image to identify civic issues.
Valid Issue Categories: {categories}

For each issue, provide:
1. Type (MUST be one of the Valid Issue Categories listed above if it fits)
2. Bounding box coordinates [ymin, xmin, ymax, xmax] scaled 0-1000.
3. Severity score from 1 (minor) to 5 (hazardous).
4. A brief technical description.

YOU MUST RESPOND ONLY WITH VALID JSON. No markdown, no preamble.
{{
  "summary": "Technical overview of the scene",
  "confidence_score": 0.95,
  "issues": [
    {{
      "type": "category_name",
      "bbox": [550, 210, 670, 390],
      "severity": 4,
      "description": "Large pothole in middle of lane..."
    }}
  ]
}}

If no issues are found, return 'issues': [] and 'summary': 'No issues detected.'"""


class Issue(BaseModel):
    type: str | None = None
    bbox: list[int] | None = Field(default=None, description="[ymin, xmin, ymax, xmax] 0-1000")
    severity: int | None = Field(default=None, ge=1, le=5)
    description: str | None = None


class QwenResponse(BaseModel):
    summary: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    issues: list[Issue] = Field(default_factory=list)


def _to_jpeg(image_bytes: bytes) -> bytes:
    """
    Convert image to JPEG.
    LM Studio's Qwen3-VL only accepts JPEG/PNG data-URLs — WebP causes
    'url field must be a base64 encoded image' or 'Invalid url' errors.
    """
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()

async def call_qwen_vision(image_bytes: bytes, mime_type: str = "image/jpeg") -> QwenResponse:
    """
    Analyse a civic image using LM Studio's local OpenAI-compatible API.
    Drop-in replacement for the previous HF / Gemini / Ollama backends.

    LM Studio accepts images as OpenAI-style base64 data-URLs.
    Since inference is local there are no external payload size limits.
    """
    settings = get_settings()

    # LM Studio Qwen3-VL rejects WebP — convert everything to JPEG first
    jpeg_bytes = _to_jpeg(image_bytes)
    b64_image = base64.b64encode(jpeg_bytes).decode()
    # LM Studio requires a proper data-URL (raw base64 alone → 'Invalid url')
    data_url = f"data:image/jpeg;base64,{b64_image}"

    # OpenAI-compatible chat/completions payload
    categories = _get_mail_categories()
    payload = {
        "model": settings.lmstudio_model,
        "stream": False,
        "temperature": 0.01,       # near-deterministic for JSON stability
        "max_tokens": 512,
        # NOTE: response_format is NOT set — LM Studio vision models reject it
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(categories=categories),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": "Inspect this image and provide a JSON report.",
                    },
                ],
            },
        ],
    }

    chat_url = f"{settings.lmstudio_base_url.rstrip('/')}/v1/chat/completions"
    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.qwen_timeout_seconds) as client:
        for attempt in range(1, settings.qwen_max_retries + 1):
            try:
                response = await client.post(chat_url, json=payload)
                response.raise_for_status()

                raw_data = response.json()
                # Standard OpenAI response shape
                content: str = (
                    raw_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                if not content:
                    raise ValueError(f"Empty response from LM Studio: {raw_data}")

                # Strip accidental markdown fences
                clean_json = (
                    content.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )

                parsed = QwenResponse.model_validate_json(clean_json)

                logger.info(
                    "lmstudio_perception_complete",
                    model=settings.lmstudio_model,
                    attempt=attempt,
                    issues_count=len(parsed.issues),
                    confidence=parsed.confidence_score,
                )
                return parsed

            except httpx.HTTPStatusError as exc:
                # Log the full response body so we can see LM Studio's error detail
                try:
                    err_body = exc.response.text
                except Exception:
                    err_body = "<unreadable>"
                last_error = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "lmstudio_perception_retry",
                    attempt=attempt,
                    status_code=exc.response.status_code,
                    error_body=err_body,
                    next_retry_in=f"{wait}s",
                )
                if attempt < settings.qwen_max_retries:
                    await asyncio.sleep(wait)

            except (httpx.RequestError, ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "lmstudio_perception_retry",
                    attempt=attempt,
                    error=str(exc),
                    next_retry_in=f"{wait}s",
                )
                if attempt < settings.qwen_max_retries:
                    await asyncio.sleep(wait)

    raise RuntimeError(
        f"LM Studio vision perception failed after {settings.qwen_max_retries} attempts. "
        f"Last error: {last_error}"
    )