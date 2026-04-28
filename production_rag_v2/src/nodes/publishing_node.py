import pprint
import os
import json
import re
import httpx
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()
from upstash_redis import Redis
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import adispatch_custom_event

# --- MCP INTEGRATION ---
from src.infra.mcp_service import (
    notion_inspect_id, 
    notion_append_notes
)
from src.services.llm_service import llm_service
from src.services.redis_service import RedisService
from src.engine.state import AgentState

FORMAT_PROMPT = """You are the Document Formatting Module.
Your goal is to transform the provided content into a professional, structured document for Notion.

CONTENT:
{content}

MODE: {mode}

INSTRUCTIONS:
1. Generate a descriptive 'title' for the document.
2. Format the 'body':
   - If Mode is RAW: Preserve the literal dialogue and content exactly; do not paraphrase or compress. 
     Act only as a layout formatter.
   - If Mode is SYNTHESIS: Refine, compress, and paraphrase the content into a cohesive, concise report. 
     Use bullet points for facts and decisions.
3. Return JSON:
{{
  "title": "Title here",
  "body": "Full formatted body here"
}}
"""

def build_raw_export(msg_ids : list[str]) -> str:
    logs = []

    for mid in msg_ids:
        data = RedisService.get_cached_conversation(mid)

        if not data:
            continue
            
        logs.append( {
            "msg_id" : mid,
            "query" : data["query"],
            "answer": data.get("payload", {}).get("answer") or 
                    data.get("payload", {}).get("initial_response", ""),
            "metadata" : data.get("metadata", {}),
            "timestamp" : data.get("timestamp")
        })
    return "\n\n---\n\n".join(
        f"USER: {l['query']}\nASSISTANT: {l['answer']}"
        for l in logs)


def _normalize_text(text: str) -> str:
    return (text or "").lower().strip()


def _is_summary_export(text: str, export_mode: str) -> bool:
    text = _normalize_text(text)
    if str(export_mode).upper() == "SYNTHESIS":
        return True
    return any(
        phrase in text for phrase in (
            "summarize our prior conversations",
            "summarize our conversations",
            "summary of our conversations",
            "synthesize our conversations",
            "compress our conversations",
        )
    )


def _collect_archived_msg_ids(user_id: str, current_msg_id: str | None = None) -> List[str]:
    ordered: list[tuple[str, str]] = []
    seen = set()

    for category in ("casual", "retrieval", "meeting"):
        for pointer in RedisService.query_index(user_id, category):
            msg_id = pointer.get("msg_id")
            if not msg_id or msg_id == current_msg_id or msg_id in seen:
                continue
            seen.add(msg_id)
            ordered.append((pointer.get("timestamp", ""), msg_id))

    ordered.sort(key=lambda x: x[0])
    return [msg_id for _, msg_id in ordered]


async def publishing_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Intelligent Publishing Agent (v2.1).
    Implements Phases 7-12: Auth -> Target -> Retrieval -> Format -> Delivery.
    """
    # --- Phase 0: Setup ---
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id", configurable.get("thread_id", "default_user"))
    redis_client = Redis(url=os.environ.get("UPSTASH_REDIS_REST_URL"), token=os.environ.get("UPSTASH_REDIS_REST_TOKEN"))
    
    msg = state.get("msg", {})
    export_mode = state.get("export_mode", "SYNTHESIS")
    # print(f"Check the memory_payload {memory_payload} \n") 
    target_meta = state.get("notion_workspace") or {}

    # --- Phase 7: Auth ---
    import asyncio
    import time
    start_time = time.time()
    raw_token = redis_client.get(f"user_token:{user_id}")
     
    if not raw_token:
        auth_url = f"https://auth.ivyllmnotion.io.vn/api/notion/oauth/connect?user_id={user_id}"
        await adispatch_custom_event("publishing_stream", {"chunk": f"✋ Auth Required: {auth_url}\n"})
        await adispatch_custom_event("publishing_stream", {"chunk": "⏳ Waiting for authentication (60s timeout)...\n"})
        
        delay = 1
        while time.time() - start_time < 60: # this code may not scale well if multiple user's send the same-request & probe the same-keys 
            key = f"user_token:{user_id}"
            # raw_token = redis_client.get(key)
            raw_token = await asyncio.to_thread(redis_client.get, key)
            if raw_token:
                await adispatch_custom_event("publishing_stream", {"chunk": "✅ Authentication successful!\n"})
                await adispatch_custom_event("publishing_stream", {"chunk": "🚀 Resuming publishing flow...\n"})
                break

            await asyncio.sleep(delay) # Non-blocking wait
            delay = min(delay * 1.5, 5)
        

        if not raw_token:
            await adispatch_custom_event("publishing_stream", {"chunk": "❌ Authentication timed out.\n"})
            return {"route": "halting"}

    # --- Phase 8: Target Resolution ---
    # Attempt to resolve from message content if a URL is present, otherwise use existing target
    current_text = msg.get("msg_content", "")
    target_id = target_meta.get("target_id")

    if "notion.so" in current_text:
        urls = re.findall(r'https?://[^\s/$.?#].[^\s]*notion\.so[^\s]*', current_text)
        if urls:
            inspection = await notion_inspect_id(urls[0], user_id)
            if inspection.get("type") in {"page", "database"} and inspection.get("id"):
                target_id = inspection["id"]
                target_meta.update({"target_id": target_id, "target_type": inspection["type"]})
            else:
                inspect_error = inspection.get("error", "Unknown Notion target error.")
                await adispatch_custom_event(
                    "publishing_stream",
                    {
                        "chunk": (
                            "❌ Notion target is inaccessible. "
                            "Make sure the page/database is in the authenticated workspace "
                            "and shared with the integration.\n"
                            f"   Details: {inspect_error}\n"
                        )
                    },
                )
                return {"route": "halting"}

    if not target_id:
        await adispatch_custom_event("publishing_stream", {"chunk": "❌ No Notion target resolved.\n"})
        return {"route": "halting"}

    if target_meta.get("target_type") == "database":
        await adispatch_custom_event(
            "publishing_stream",
            {
                "chunk": (
                    "❌ Export target is a Notion database, but this flow only appends blocks to pages. "
                    "Use a page target or implement database row export for this route.\n"
                )
            },
        )
        return {"route": "halting"}

    # --- Phase 9: Export Source Selection ---
    raw_content = ""
    current_msg_id = msg.get("msg_id")
    try:
        archived_msg_ids = _collect_archived_msg_ids(user_id, current_msg_id=current_msg_id)

        if not _is_summary_export(current_text, export_mode):
            raw_content = build_raw_export(archived_msg_ids)
            if not raw_content:
                raw_content = "No prior archived conversations were found for export."
        else:
            episodes_raw = redis_client.lrange(f"memory:{user_id}", 0, 9)
            episodes = []
            for e in episodes_raw:
                if not e:
                    continue
                if isinstance(e, bytes):
                    e = e.decode()
                try:
                    episodes.append(json.loads(e))
                except Exception:
                    continue

            candidates = []
            for e in episodes:
                summary = json.dumps(e.get("payload", {}))
                for mid in e.get("msg_ids", []):
                    candidates.append({"msg_id": mid, "summary": summary})

            filtered = [
                c for c in candidates
                if any(word in c["summary"].lower() for word in current_text.lower().split())
            ]
            if not filtered:
                filtered = candidates[:20]

            target_ids = []
            if filtered:
                ranking_prompt = f"""
                You are a ranking engine.

                USER QUERY:
                {current_text}

                CANDIDATES:
                {json.dumps(filtered[:30], indent=2)}

                Select the MOST relevant msg_ids (max 10).

                Return ONLY JSON list:
                ["msg_1", "msg_2"]
                """
                res = await llm_service.generate(ranking_prompt)
                clean = res.strip().replace("```json", "").replace("```", "")
                try:
                    ranked_ids = json.loads(clean)
                except Exception:
                    ranked_ids = [c["msg_id"] for c in filtered[:10]]

                valid_ids = {c["msg_id"] for c in candidates}
                target_ids = [mid for mid in ranked_ids if mid in valid_ids][:10]

            if not target_ids:
                target_ids = archived_msg_ids[-10:]

            fetched_logs = []
            for tid in target_ids:
                if not tid:
                    continue
                raw_turn = RedisService.get_cached_conversation(tid)
                if raw_turn:
                    fetched_logs.append({
                        "query": raw_turn.get("query"),
                        "answer": raw_turn.get("payload", {}).get("answer") or raw_turn.get("payload", {}).get("initial_response", "")
                    })
            if fetched_logs:
                logs_text = "\n\n".join([f"Q: {l['query']}\nA: {l['answer']}" for l in fetched_logs])
                synth_prompt = f"### TASK: Forge a final status report from these RAW CONVERSATION LOGS.\n\nLOGS:\n{logs_text}\n\nReturn professional Markdown."
                raw_content = await llm_service.generate(synth_prompt)
            else:
                raw_content = "No prior conversations were available to summarize."

    except Exception as e:
        print(f"⚠️ [Publisher] Selector Agent Failed by Error: {e}")
        raw_content = current_text

    # --- Phase 10: Formatting (LLM Polishing) ---
    await adispatch_custom_event("publishing_stream", {"chunk": f"🎨 Formatting content ({export_mode})...\n"})
    format_prompt = FORMAT_PROMPT.format(content=raw_content, mode=export_mode)
    formatted_res = await llm_service.generate(format_prompt)
    try:
        doc = json.loads(formatted_res)
    except:
        doc = {"title": "Exported Note", "body": raw_content}
    # --- Phase 11 & 12: Block Mapping & Delivery ---
    await adispatch_custom_event("publishing_stream", {"chunk": f"🚀 Shipping '{doc['title']}' to Notion...\n"})
    
    # We combine title and body for the final block append
    final_text = f"# {doc['title']}\n\n{doc['body']}"

    # --- Phase 10b: File Logging (Separate Log-File) fot DEBUGING ---
    try:
        log_dir = os.path.join(os.getcwd(), "production_rag_v2", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "publishing_debug.log")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] 🔍 DEBUG: Finalizing export payload for '{doc['title']}'\n")
            f.write(f"📝 CONTENT PREVIEW:\n{final_text}\n")
            f.write("="*80 + "\n\n")
    except Exception as log_e:
        # Fallback to custom event only if file write fails
        await adispatch_custom_event("publishing_stream", {"chunk": f"⚠️ Log Write Failed: {str(log_e)}\n"})
    
    try:
        res = await notion_append_notes(
            block_id=target_id,
            content=final_text,
            user_id=user_id
        )
        if "error" in res:
            raise Exception(res["error"])
        
        final_msg = f"✅ Export Complete: '{doc['title']}' sent to Notion."
        await adispatch_custom_event("publishing_stream", {"chunk": f"{final_msg}\n"})
        
        return {
            "msg": {**msg, "msg_content": final_msg},
            "route": "halting",
            "notion_workspace": target_meta
        }
    except Exception as e:
        await adispatch_custom_event("publishing_stream", {"chunk": f"❌ Export Failed: {str(e)}\n"})
        return {"route": "halting"}
