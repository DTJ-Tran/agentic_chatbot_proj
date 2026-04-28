import json
import asyncio
import time
import logging
import numpy as np
from typing import Dict, Any
from upstash_redis import Redis
from src.core.config import settings
from src.services.search_service import SearchService


def _json_safe(obj):
    """Recursively convert non-JSON-serializable types (numpy, etc.) to Python natives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class QueueWorker:
    """
    Clean async worker with:
    - Redis queue (task source)
    - Future registry (direct response path)
    """

    _futures: Dict[str, asyncio.Future] = {}

    @classmethod
    def register_future(cls, msg_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        cls._futures[msg_id] = future
        return future

    @classmethod
    def resolve_future(cls, msg_id: str, data: Any):
        future = cls._futures.pop(msg_id, None)
        if future and not future.done():
            future.set_result(data)

    def __init__(self):
        from src.services.redis_service import redis_service
        self.redis = redis_service.get_client()
        self.search_service = SearchService()
        self.queue_key = "search_query_queue"
        self._stop_event = asyncio.Event()
        self.logger = logging.getLogger(__name__)

    def stop(self):
        """Signals the worker to stop polling."""
        self._stop_event.set()

    async def poll_once(self):
        loop = asyncio.get_running_loop()

        # 1. Pop task
        try:
            raw_task = await loop.run_in_executor(
                None, lambda: self.redis.rpop(self.queue_key)
            )
        except Exception as e:
            self.logger.warning("[Worker] Redis connection error: %s", e)
            return False

        if not raw_task:
            return False

        msg_id = None
        try:
            task = json.loads(raw_task)
            msg_id = task.get("msg_id")
            query = task.get("usr_content")

            if not msg_id:
                self.logger.warning("[Worker] Found malformed task with no msg_id.")
                return True

            if settings.debug:
                self.logger.info("[Worker] Began processing task: %s", msg_id)
                self.logger.info("[Worker] Query: %s", query[:50] if query else "")
            
            # Executing heavy pipeline
            raw_results = await self.search_service.run_pipeline(query)
            
            # ✅ Convert numpy types (float64, int64) to native Python types
            results = _json_safe(raw_results)

            payload = {
                "msg_id": msg_id,
                "search_res": results,
                "status": "resolved",
                "resolved_at": time.time(),
            }

            # ✅ Persist
            if settings.debug:
                self.logger.info("[Worker] Persisting %s docs to Redis.", len(results))
            await loop.run_in_executor(
                None, lambda: self.redis.set(msg_id, json.dumps(payload), ex=3600)
            )
            if settings.debug:
                self.logger.info("[Worker] Persisted results to Redis key: %s", msg_id)
            
            # ✅ Resolve
            QueueWorker.resolve_future(msg_id, results)

            if settings.debug:
                self.logger.info("[Worker] Finished %s (%s docs)", msg_id, len(results))
            return True

        except Exception as e:
            self.logger.exception("[Worker] Failure processing %s: %s", msg_id, e)
            # Ensure synthesis doesn't hang indefinitely
            if msg_id:
                QueueWorker.resolve_future(msg_id, [])
            return True

    async def flush_stale_queue(self):
        """Discards leftover tasks from previous sessions on startup."""
        loop = asyncio.get_running_loop()
        flushed = 0
        while True:
            item = await loop.run_in_executor(
                None, lambda: self.redis.rpop(self.queue_key)
            )
            if item is None:
                break
            flushed += 1
        if flushed:
            self.logger.info("[Worker] Flushed %s stale task(s) from previous session.", flushed)

    async def start(self):
        self.logger.info("[Worker] Polling loop started.")
        await self.flush_stale_queue()
        
        last_heartbeat = time.time()
        while not self._stop_event.is_set():
            found = await self.poll_once()
            
            # Print heartbeat every 60s if idle
            if not found and (time.time() - last_heartbeat > 60):
                # print(f"💓 [Worker] Heartbeat: Still polling Redis '{self.queue_key}'...")
                last_heartbeat = time.time()
            elif found:
                last_heartbeat = time.time()

            await asyncio.sleep(0.1 if found else 1.0)
