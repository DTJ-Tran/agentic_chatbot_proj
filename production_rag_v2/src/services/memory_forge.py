import json
import os
import time
import asyncio
import logging
from datetime import datetime
from upstash_redis import Redis
from src.services.llm_service import llm_service

MEMORY_FORGE_PROMPT = """You are the 'Memory Forge' background worker.
Your task is to synthesize raw conversation turns into a coherent 'Episodic Memory'.

CONTEXT (PRIOR EPISODES):
{prior_episodes}

NEW RAW CONTENT TO FUSE:
{new_content}

INSTRUCTIONS:
1. Review the Prior Episodes for continuity (names, unresolved topics, established facts).
2. Synthesize the New Raw Content into a compact 'Fact Sheet' (JSON).
3. Ensure the new episode addresses any 'unresolved' items from the prior context if they were settled in this turn.
4. Focus on: Entities, Decisions, and New Facts.

Return ONLY valid JSON:
{{
  "facts": ["..."],
  "entities": ["..."],
  "decisions": ["..."],
  "unresolved": ["..."],
}}
"""

class MemoryForge:
    def __init__(self):
        self.redis = Redis(
            url=os.environ.get("UPSTASH_REDIS_REST_URL"), 
            token=os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        )
        self.queue_name = "memory_refinery_queue"
        self.ep_queue_name = "memory"
        self._stop_event = asyncio.Event()
        self.logger = logging.getLogger(__name__)

    def stop(self):
        """Signals the background loop to stop polling."""
        self._stop_event.set()

    async def start(self):
        self.logger.info("[Memory Forge] Background worker started. Listening for raw memories.")
        # Use a semaphore to allow 3 concurrent LLM synthesis tasks
        # This resolves the bottleneck by not waiting for one turn to finish before starting the next

        from collections import deque, defaultdict
        buffers = defaultdict(lambda: deque(maxlen=5)) # buffer being partition across user
        # buffer = deque(maxlen=5) 
        step_counters = defaultdict(int)
        STEP =3
        semaphore = asyncio.BoundedSemaphore(3) # introduce the semaphore (a counter for LLM-calls)
        locks = defaultdict(asyncio.Lock)
        async def process_item(raw_datas: list[dict[str, object]]):
                try:
                    first_item = raw_datas[0]
                    if isinstance(first_item, str):
                        first_item = json.loads(first_item)
                    user_id = first_item.get("user_id") # because all share the identical user_id
                    async with locks[user_id]: # only 1 user at a time & order gurantee
                            
                            epsisode_text = f"user_id : {user_id}"
                            intents = []
                            msg_ids = []
                            for raw_data in raw_datas:
                                item = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                                # raws.extend(item)
                                # user_id = item.get("user_id") # get the user_id
                                msg_id = item.get("msg_id")
                                new_content = item.get("content")
                                intent = item.get("intent")
                                
                                if intent:
                                    intents.append(intent)
                                if msg_id:
                                    msg_ids.append(msg_id)
                                epsisode_text += f"\n conversation_id {msg_id}: msg_content {new_content} : msg_intent {intent}"
                                
                                # print(f"🔨 [Memory Forge] Processing {msg_id} for {user_id}...")

                            # 2. Get the last-episode
                            mem_key = f"{self.ep_queue_name}:{user_id}" # building the episode keys

                            last_episode = self.redis.lrange(mem_key, start=0, stop=0) # take the latest element in the episodic queue
                            prior_text = "[]"
                            if last_episode:
                                ep = json.loads(last_episode[0])
                                payload = ep.get("payload", {})
                                facts = payload.get("facts", [])
                                entities = payload.get("entities", [])
                                decisions = payload.get("decisions", [])
                                unresolved = payload.get("unresolved", [])
                                prior_text = f"""
                                Facts: {facts}
                                Entities: {entities}
                                Decision: {decisions}
                                Unresolved: {unresolved}
                                """
                               
                            # 3. Forge the New Episode
                            prompt = MEMORY_FORGE_PROMPT.format(prior_episodes=prior_text, new_content=epsisode_text)
                            async with semaphore: # semaphore -> lock # only N thing at a time for n-task from 1 user
                                refined_json_str = await llm_service.generate(prompt)

                            # Sanitization
                            if "```json" in refined_json_str:
                                refined_json_str = refined_json_str.split("```json")[1].split("```")[0]
                            
                            payload = json.loads(refined_json_str.strip())
                            
                            # 4. Push and Anchor
                            episode = {
                                "ts": datetime.now().isoformat(),
                                "msg_ids": msg_ids, # THE CRITICAL LINK
                                "intents": intents,
                                "payload": payload,
                            }
                            
                            self.redis.lpush(mem_key, json.dumps(episode))
                            self.redis.ltrim(mem_key, 0, 49) # Keep top 50 
                            self.redis.expire(mem_key, 604800) # 7-Day TTL for episodic memory
                            
                            self.logger.info("[Memory Forge] Success: %s forged and anchored.", msg_id)

                except Exception as e:
                    self.logger.exception("[Memory Forge] Item failed: %s", e)

        while not self._stop_event.is_set():
            raw_data = None
            try:
                # this code drop dues to the history-erasure & non-deterministicity & Redis was controlling window
                # 0. check the size of the queue -> if size_queue > 5 -> pop 3 first element & remain 2 last elements 
                # raw_datas= None
                # if self.redis.llen(self.queue_name) >= 5:
                #     pipeline = self.redis.multi()
                #     pipeline.lpop(self.queue_name, 3) # a list of str
                #     pipeline.lrange(self.queue_name, start=0, stop=1) # read 2 first elements
                #     results = pipeline.exec()
                #     popped , remaining_top = results[0], results[1]
                #     raw_datas = popped + remaining_top

                raw_data = self.redis.rpop(self.queue_name)
                if raw_data:
                    preview = raw_data if isinstance(raw_data, str) else str(raw_data)
                    self.logger.info(
                        "[Memory Forge] Dequeued raw_data from %s: %s",
                        self.queue_name,
                        preview[:200],
                    )
                else:
                    self.logger.debug("[Memory Forge] No raw_data available in %s", self.queue_name)

                if raw_data: # worker memory control window
                    """"
                    A1 → buffer [A1]
                    A2 → [A1,A2]
                    A3 → [A1,A2,A3]
                    A4 → [A1,A2,A3,A4]
                    A5 → [A1,A2,A3,A4,A5] → step=1
                    A6 → [A2,A3,A4,A5,A6] → step=2
                    A7 → [A3,A4,A5,A6,A7] → step=3 → emit        
                    """
                    item = json.loads(raw_data)
                    user_id = item.get("user_id")
                    buf = buffers[user_id]
                    buf.append(item)
                    self.logger.info("[Memory Forge] Buffered item for user=%s size=%s", user_id, len(buf))
                    # buffer.append(item)
                    if len(buf) == 5:
                        step_counters[user_id] +=1
                        # Emit immediately on first full window
                        if step_counters[user_id] == 1 or (step_counters[user_id] -1) % STEP == 0:
                            window = list(buf)
                            self.logger.info(
                                "[Memory Forge] Scheduling forge for user=%s step=%s window_msg_ids=%s",
                                user_id,
                                step_counters[user_id],
                                [w.get("msg_id") for w in window],
                            )
                            # step_counters[user_id] = 0 
                            # using per-use lock
                            asyncio.create_task(process_item(window)) # this not gurantee if the episodic are being created in Order
                
                # Launch concurrent task
                # Compress all 5 conversation into 1 episodic memory
                # asyncio.create_task(process_item(raw_datas))

            except Exception as e:
                self.logger.exception("[Memory Forge] Loop error: %s", e)
                await asyncio.sleep(5)

            if not raw_data:
                await asyncio.sleep(1.0)


forge = MemoryForge()
