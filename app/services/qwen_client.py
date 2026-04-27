"""
Client for Qwen3-VL vision API.
Prompts the model to return a structured JSON report for municipal issues.
Implements retry with exponential backoff and strict Pydantic validation.
"""
import asyncio
import base64
import json
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert AI Municipal Inspector. 
Analyze the image to identify civic issues (potholes, garbage, water leaks, etc.).

For each issue, provide:
1. Category (e.g., 'Road Infrastructure', 'Sanitation')
2. Bounding box coordinates [ymin, xmin, ymax, xmax] scaled 0-1000.
3. Severity score from 1 (minor) to 5 (hazardous).
4. A brief technical description.

YOU MUST RESPOND ONLY WITH VALID JSON. No markdown, no preamble.
{
  "summary": "Technical overview of the scene",
  "overall_confidence": 0.95,
  "issues": [
    {
      "type": "pothole",
      "bbox": [550, 210, 670, 390],
      "severity": 4,
      "description": "Large pothole in middle of lane, hazard to motorcyclists."
    }
  ]
}

If no issues are found, return 'issues': [] and 'summary': 'No issues detected.'"""

class Issue(BaseModel):
    type: str
    bbox: list[int] = Field(..., description="[ymin, xmin, ymax, xmax] 0-1000")
    severity: int = Field(..., ge=1, le=5)
    description: str

class QwenResponse(BaseModel):
    summary: str
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    issues: list[Issue]

async def call_qwen_vision(image_bytes: bytes, mime_type: str = "image/jpeg") -> QwenResponse:
    """
    Call Qwen3-VL via Hugging Face Inference API. 
    Orchestrates Phase 2 of the Civic-Duty Agent pipeline.
    """
    settings = get_settings()
    b64_image = base64.b64encode(image_bytes).decode()

    # Refined payload for HF Chat/Vision completion
    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "system", 
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                    {
                        "type": "text", 
                        "text": "Inspect this image and provide a JSON report."
                    },
                ],
            }
        ],
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.01, # Near-deterministic for JSON stability
            "return_full_text": False
        },
    }

    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.qwen_timeout_seconds) as client:
        for attempt in range(1, settings.qwen_max_retries + 1):
            try:
                response = await client.post(
                    settings.qwen_api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {settings.huggingface_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                
                # Handle HF specific status codes (e.g., 503 Model Loading)
                if response.status_code == 503:
                    raise httpx.HTTPStatusError("Model is currently loading", request=response.request, response=response)
                
                response.raise_for_status()
                raw_data = response.json()

                # Robust parsing of HF output formats
                content = ""
                if isinstance(raw_data, list) and len(raw_data) > 0:
                    content = raw_data[0].get("generated_text", "")
                elif isinstance(raw_data, dict):
                    # Check for OpenAI-style or standard HF dict
                    content = raw_data.get("choices", [{}])[0].get("message", {}).get("content") or raw_data.get("generated_text", "")

                if not content:
                    raise ValueError(f"Empty response from API: {raw_data}")

                clean_json = content.strip().removeprefix("```json").removesuffix("```").strip()
                
                parsed = QwenResponse.model_validate_json(clean_json)
                
                logger.info(
                    "qwen_perception_complete",
                    attempt=attempt,
                    issues_count=len(parsed.issues),
                    confidence=parsed.overall_confidence,
                )
                return parsed

            except (httpx.HTTPStatusError, httpx.RequestError, ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "qwen_perception_retry",
                    attempt=attempt,
                    error=str(exc),
                    next_retry_in=f"{wait}s"
                )
                if attempt < settings.qwen_max_retries:
                    await asyncio.sleep(wait)

    raise RuntimeError(
        f"Qwen3-VL Perception failed after {settings.qwen_max_retries} attempts. Last error: {last_error}"
    )