import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from src.services.redis_service import redis_service
from src.core.config import settings
from src.services.vector_service import VectorService
from src.services.queue_worker import QueueWorker

def _format_search_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Search completed but no relevant web results were found."
    
    formatted = []
    for i, res in enumerate(results):
        url = res.get("url", "")
        content = res.get("full_content", "") or "\n".join(res.get("snippets", []))
        formatted.append(f"EXT-[{i+1}] {content}\n(URL: {url})")
    
    return "\n\n".join(formatted)

@tool
async def db_retrieval_tool(query: str) -> str:
    """
    Searches the internal FPT Software policy database for relevant documents.
    Use this tool when the user asks about company policies, benefits, or internal procedures.
    """
    vector_service = VectorService()
    try:
        # Implementation of the 'time-out trick' to prevent blocking the agent loop
        results = await asyncio.wait_for(
            vector_service.search(query=query, limit=5),
            timeout=3.0
        )
        
        if not results:
            return "No relevant internal policy documents found."
        
        formatted_results = []
        for i, res in enumerate(results):
            content = res.get("content", "")
            metadata = res.get("metadata", {})
            formatted_results.append(f"[{i+1}] {content}\n(Source: {metadata.get('source', 'Unknown')})")
        
        return "\n\n".join(formatted_results)
    
    except asyncio.TimeoutError:
        print(f"⚠️ [DB Tool] Search timed out for query: {query}")
        return (
            "Internal DB search timed out. Moving to fallback. "
            "Please check external results via redis_retrieval_tool to see if information is available there."
        )
    except Exception as e:
        print(f"❌ [DB Tool] Error: {e}")
        return f"Error retrieving policy documents: {e}"

@tool
async def redis_retrieval_tool(msg_id: str) -> str:
    """
    Polls the Redis cache for external search results using a msg_id.
    Use this tool when external market data or latest info from the internet is required.
    Wait for up to 10 seconds for the results to appear.
    """
    redis = redis_service.get_client()
    timeout = 10.0
    start_time = time.time()
    
    # Fast path: Check future registry
    future = QueueWorker._futures.get(msg_id)
    if future:
        try:
            results = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            # Future resolved, but we still pop from registry
            QueueWorker._futures.pop(msg_id, None)
            return _format_search_results(results)
        except asyncio.TimeoutError:
            print(f"⏳ [Redis Tool] Future timed out after {timeout}s → checking Redis direct.")

    # Slow path: Redis polling
    while (time.time() - start_time) < timeout:
        cached = await asyncio.get_event_loop().run_in_executor(
            None, lambda: redis.get(msg_id)
        )
        if cached:
            data = json.loads(cached)
            results = data.get("search_res", [])
            if results:
                return _format_search_results(results)
        await asyncio.sleep(0.5)

    return "No external search results found after 10s wait."
