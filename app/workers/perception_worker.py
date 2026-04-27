"""
Perception worker: long-running async process that reads report IDs
from Redis and runs the perception pipeline for each.

Run with:
    python -m app.workers.perception_worker
"""

import asyncio
import signal 
import uuid
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.redis import get_redis_pool
from app.db.session import AsyncSessionLocal, engine
from app.services.perception import run_perception
import redis.asyncio as aioredis

logger=get_logger(__name__)
_shutdown=False

def _handle_signal(sig, frame):
    global _shutdown
    logger.info("worker_shutdown_signal", signal=sig)
    _shutdown=True

async def _fetch_image_bytes(report_id: uuid.UUID) -> tuple[bytes, str] | None:
    import os
    path=f"/tmp/images/{report_id}"
    for ext, mime in [(".jpg", "image/jpeg"), (".png", "image/png"), (".webp", "image/webp"), (".jpeg", "image/jpeg")]:
        full_path = path+ext 
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                return f.read(), mime
    logger.error("worker_image_not_found", report_id=str(report_id))
    return None

async def process_one(report_id_str: str, redis_client: aioredis.Redis) -> None:
    try:
        report_id = uuid.UUID(report_id_str)
    except ValueError:
        logger.error("worker_invalid_report_id", value=report_id_str)
        return

    image_data = await _fetch_image_bytes(report_id)
    if not image_data:
        return
    
    image_bytes, mime_type = image_data

    async with AsyncSessionLocal() as db:
        await run_perception(
            report_id=report_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            db=db,
            redis=redis_client
        )

async def run_worker() -> None:
    settings=get_settings()
    setup_logging()
    logger.info("perception_woker_started", concurrency=settings.perception_worker_concurrency)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pool=get_redis_pool()
    redis_client=aioredis.Redis(connection_pool=pool)
    semaphore=asyncio.Semaphore(settings.perception_worker_concurrency)

    async def bounded_process(item: str):
        async with semaphore:
            await process_one(item, redis_client)

    try:
        while not _shutdown:
            result = await redis_client.blpop(settings.perception_queue_key, timeout=2)
            if result is None:
                continue
            _, report_id_bytes=result
            report_id_str=report_id_bytes if isinstance(report_id_bytes, str) else report_id_bytes.decode()
            logger.info("worker_dequeued", report_id=report_id_str)
            asyncio.create_task(bounded_process(report_id_str))
    finally:
        await redis_client.aclose()
        await engine.dispose()
        logger.info("perception_worker_stopped")

if __name__ == "__main__":
    asyncio.run(run_worker())