"""
CCTV Worker: background task to pull frames from RTSP streams.
"""
import asyncio
import cv2
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.redis import get_redis_pool
from app.db.session import AsyncSessionLocal
from app.services.stream_service import process_frame_logic

logger = get_logger(__name__)

# Shared state for background CCTV tasks
active_cctv_tasks = {}

async def cctv_worker_task(rtsp_url: str, lat: float | None, lon: float | None, label: str):
    """Loop to pull frames from CCTV."""
    logger.info("cctv_worker_starting", url=rtsp_url, label=label)
    
    cap = cv2.VideoCapture(rtsp_url)
    try:
        while rtsp_url in active_cctv_tasks:
            ret, frame = cap.read()
            if not ret:
                logger.warning("cctv_frame_failed", url=rtsp_url)
                cap.release()
                await asyncio.sleep(10)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Capture frame every 15 seconds
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                image_bytes = buffer.tobytes()
                # Run logic in a new session
                async with AsyncSessionLocal() as db:
                    pool = get_redis_pool()
                    async with aioredis.Redis(connection_pool=pool) as redis:
                        await process_frame_logic(image_bytes, db, redis, lat, lon, label)

            await asyncio.sleep(15)
    except Exception as e:
        logger.error("cctv_worker_error", url=rtsp_url, error=str(e))
    finally:
        cap.release()
        active_cctv_tasks.pop(rtsp_url, None)
        logger.info("cctv_worker_stopped", url=rtsp_url)


def start_cctv_task(rtsp_url: str, lat: float | None, lon: float | None, label: str):
    """Entry point to start a new CCTV task."""
    if rtsp_url in active_cctv_tasks:
        return False
    
    task = asyncio.create_task(cctv_worker_task(rtsp_url, lat, lon, label))
    active_cctv_tasks[rtsp_url] = task
    return True


def stop_cctv_task(rtsp_url: str):
    """Stops a running CCTV task."""
    if rtsp_url in active_cctv_tasks:
        active_cctv_tasks.pop(rtsp_url)
        return True
    return False
