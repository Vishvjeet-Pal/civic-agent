import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.exif import extract_exif, ExifData
from app.services.qwen_client import QwenResponse, Issue

def test_exif_no_data_returns_none_gracefully():
    minimal_jpeg=bytes([0xFF, 0xD8, 0xFF,0xE0] + [0x00]*100)
    result=extract_exif(minimal_jpeg)
    assert result==ExifData(None,None,None)

def test_exif_corrupt_data_does_not_raise():
    result=extract_exif(b"not_an_image_at_all")
    assert result.latitude is None
    assert result.longitude is None

@pytest.mark.asyncio 
async def test_qwen_client_retries_on_failure():
    with patch("app.services.qwent_client.httpx.AsyncClient") as mock_client_cls:
        mock_client=AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value=mock_client

        from app.services.qwen_client import call_qwen_vision 
        with pytest.raises(RuntimeError, match="failed after"):
            await call_qwen_vision(b"fake_image_bytes")
        
        assert mock_client.post.call_count==3

def test_qwen_response_validation():
    valid = QwenResponse(
        summary="Road damage detected",
        confidence_score=0.91,
        issues=[
            Issue(
                type="pothole",
                bbox=[100,200,300,400],
                severity=4,
                description="Large pothole"
            )
        ]
    )
    assert valid.confidence_score == 0.91
    assert len(valid.issues)==1

def test_low_confidence_response():
    result = QwenResponse(
        summary="Possible graffiti detected",
        confidence_score=0.60,
        issues=[
            Issue(
                type="graffiti",
                bbox=[10, 20, 30, 40],
                severity=2,
                description="Graffiti on public wall"
            )
        ]
    )

    assert result.confidence_score < 0.75