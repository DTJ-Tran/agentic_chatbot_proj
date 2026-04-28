import json
import os
import logging
from datetime import datetime
from typing import Dict, Any

from src.engine.state import AgentState
from upstash_redis import Redis
from src.services.llm_service import llm_service
from src.services.redis_service import RedisService
from src.services.intent_module import IntentModule
from src.services.mem_raid_controller import MemRaidController

def scribe_key(msg_id: str) -> str:
    return f"scribed:{msg_id}"

class ScribingNode:
    """
    Intelligent Memory Controller (v2.2).
    Role:
    - Classify intent
    - Decide whether to store
    - Emit atomic events to MemoryForge
                    ┌──────────────┐
                    │ ScribingNode │
                    └──────┬───────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
    memory_refinery_queue         publishing_conv
    (atomic events)               (rich context)
            │                             │
            ▼                             ▼
    MemoryForge                  Notion Export
    (episodic memory)         (raw or synthesized)
    """
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)
        msg = state.get("msg", {})
        query = msg.get("msg_content", "")
        user_id = state.get("session_id", "default_user")
        msg_id = state.get("msg", {}).get("msg_id")
        redis_client = Redis(url=os.environ.get("UPSTASH_REDIS_REST_URL"), token=os.environ.get("UPSTASH_REDIS_REST_TOKEN"))

        scribe_key = f"scribed:{msg_id}"

        # IDENTITY_GUARD (already scribed)
        if redis_client.get(scribe_key):
            return {
                **state,
                "route" : state.get("route", "halting"),
                "memory_payload" : None
            }
        
        # --- Phase 1: Intent Classification ---
        intent = await IntentModule.classify(query) # Classifies the user intent based on the query and provides signals for memory/export.
        
        # ALWAYS store raw_conv_log (for persistent) - this is difference with conv_archive (a full checkpoint of the whole conversation)
        redis_client.lpush(
            f"raw_events:{user_id}",
            json.dumps({
                "msg_id": msg_id,
                "role": "user",
                "content": query,
                "ts": datetime.now().isoformat()
            })
        )
        # --- Phase 2: Memory Decision (Gatekeeping only) ---
        memory_decision = MemRaidController.evaluate(intent)
        
        # --- Phase 3: Export Decision ---
        is_export = intent.get("export_signal", False)

        store_memory = memory_decision["store_memory"]
        # export = memory_decision["exports"]
        # --- Phase 4:Push ATOMIC event to MemoryForge ---
        if store_memory:
            try:
                refinery_item = {
                    "user_id": user_id,
                    "msg_id": state.get("msg", {}).get("msg_id"),
                    "content": query,
                    "intent": intent, # richer knowledge (a dictionary)
                    "ts": datetime.now().isoformat()
                }
                redis_client.lpush(f"memory_refinery_queue", json.dumps(refinery_item))
                logger.info(
                    "[Scribing] Enqueued memory refinery item for user=%s msg_id=%s",
                    user_id,
                    refinery_item["msg_id"],
                )
            except Exception as e:
                print(f"⚠️ [Scribing] Queue Push Failed: {e}")
        else:
            logger.info(
                "[Scribing] Skipped memory enqueue for user=%s msg_id=%s category=%s score=%s export_signal=%s",
                user_id,
                msg_id,
                intent.get("category"),
                intent.get("importance_score"),
                intent.get("export_signal"),
            )

        # --- Phase 5 & 6: State Update & Export Decision ---
        route = "publishing" if is_export else "halting"
        
        # MARK AS DONE (critical)
        redis_client.set(scribe_key, "1") # TRUE
        return {
            "intent_data": intent,
            "export_mode": intent.get("export_mode", "SYNTHESIS"),
            "memory_payload": {"body" : query}, # minial payload
            "route": route,
            "publishing_conv": {
                "pending_export": {
                    # RAW_EPXORT - use query dirrectly
                    # "content": raw_export_string if intent.get("export_mode") == "RAW" else query,
                    "status": "pending"
                }
            }
        }




scribing_node = ScribingNode()
