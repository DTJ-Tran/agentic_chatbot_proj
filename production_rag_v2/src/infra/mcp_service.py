import os
import re
from dotenv import load_dotenv

load_dotenv()
import json
import httpx
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
from upstash_redis import Redis

# Initialize FastMCP - Pure Remote SSE Node
mcp = FastMCP("Notion_Remote_Node")

# Constants
NOTION_API_VERSION = "2022-06-28"
NOTION_TIMEOUT = 30.0

# Redis Setup - Must be reachable by this machine
redis_client = Redis(
    url=os.environ.get("UPSTASH_REDIS_REST_URL"),
    token=os.environ.get("UPSTASH_REDIS_REST_TOKEN"),
)

async def get_notion_headers(user_id: str) -> Dict[str, str]:
    """Fetches identity from shared cache."""
    raw_token = redis_client.get(f"user_token:{user_id}")
    if not raw_token:
        print(f"❌ AUTH ERROR: Identity '{user_id}' not found in Redis.")
        raise ValueError(f"Identity '{user_id}' not found in Redis. Please authenticate.")
    
    try:
        token_data = json.loads(raw_token)
        access_token = token_data.get("access_token", raw_token)
    except:
        access_token = raw_token

    return {
        "Authorization": f"Bearer {access_token}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }

# --- Infrastructure ---

@mcp.tool()
async def bootstrap_workspace(parent_page_id: str, user_id: str) -> Dict[str, Any]:
    """Creates 'Meeting History' and 'Q&A Notes' databases if missing."""
    # Sanitize ID: Extract 32-char hex string from potential URLs or IDs with dashes
    parent_page_id = str(parent_page_id or "").strip()
    match = re.search(r"([a-f0-9]{32})", parent_page_id.replace("-", ""))
    clean_parent_id = match.group(1) if match else parent_page_id.replace("-", "")

    print(f"🛠️  BOOTSTRAP: user={user_id} parent={clean_parent_id}")
    headers = await get_notion_headers(user_id)

    results = {"meeting_db": None, "qa_db": None, "errors": []}
    
    meeting_schema = {"Name": {"title": {}}, "Date": {"date": {}}, "Summary": {"rich_text": {}}}
    qa_schema = {"Question": {"title": {}}, "Answer": {"rich_text": {}}}
    
    async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
        # Create Meeting History
        res1 = await client.post("https://api.notion.com/v1/databases", headers=headers, json={
            "parent": {"type": "page_id", "page_id": clean_parent_id},
            "title": [{"type": "text", "text": {"content": "Meeting History"}}],
            "properties": meeting_schema
        })
        if res1.status_code == 200: 
            results["meeting_db"] = res1.json()["id"]
        else:
            results["errors"].append(f"Meeting DB Creation Error: {res1.text}")
        
        # Create Q&A Notes
        res2 = await client.post("https://api.notion.com/v1/databases", headers=headers, json={
            "parent": {"type": "page_id", "page_id": clean_parent_id},
            "title": [{"type": "text", "text": {"content": "Q&A Notes"}}],
            "properties": qa_schema
        })
        if res2.status_code == 200: 
            results["qa_db"] = res2.json()["id"]
        else:
            results["errors"].append(f"QA DB Creation Error: {res2.text}")
    
    print(f"CHECK THE RESULTS {results}")
    return results

# --- Export Tools ---

@mcp.tool()
async def notion_append_notes(block_id: str, content: str, user_id: str) -> Dict[str, Any]:
    """Export Meeting Notes / Markdown to a page."""
    print(f"📝 APPEND_NOTES: user={user_id} target={block_id}")
    headers = await get_notion_headers(user_id)
    blocks = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": line.strip()}}]}} 
              for line in content.split("\n") if line.strip()]
    
    async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
        res = await client.patch(f"https://api.notion.com/v1/blocks/{block_id}/children", headers=headers, json={"children": blocks})
        res.raise_for_status()
        return res.json()

@mcp.tool()
async def log_qa_to_notion(db_id: str, question: str, answer: str, user_id: str, prop_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Log a Q&A pair to a Notion database with dynamic property mapping.
    """
    try:
        # Use provided prop_map or fallback to defaults
        q_prop = (prop_map or {}).get("question", "Question")
        a_prop = (prop_map or {}).get("answer", "Answer")

        payload = {
            "parent": {"database_id": db_id},
            "properties": {
                q_prop: {"title": [{"text": {"content": question}}]},
                a_prop: {"rich_text": [{"text": {"content": answer}}]}
            }
        }
        async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
            headers = await get_notion_headers(user_id)
            res = await client.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
            return res.json()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def log_meeting_to_notion(db_id: str, title: str, summary: str, date: str, user_id: str, prop_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Log a meeting note to a Notion database with dynamic property mapping.
    """
    try:
        t_prop = (prop_map or {}).get("title", "Name")
        d_prop = (prop_map or {}).get("date", "Date")
        s_prop = (prop_map or {}).get("summary", "Summary")

        payload = {
            "parent": {"database_id": db_id},
            "properties": {
                t_prop: {"title": [{"text": {"content": title}}]},
                d_prop: {"date": {"start": date}},
                s_prop: {"rich_text": [{"text": {"content": summary}}]}
            }
        }
        async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
            headers = await get_notion_headers(user_id)
            res = await client.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
            return res.json()
    except Exception as e:
        return {"error": str(e)}

# --- Search ---

@mcp.tool()
async def notion_search(query: str, user_id: str) -> List[Dict[str, Any]]:
    """Search for pages and databases by title."""
    print(f"🔍 SEARCH: user={user_id} query='{query}'")
    headers = await get_notion_headers(user_id)
    payload = {
        "query": query
    }
    async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
        res = await client.post("https://api.notion.com/v1/search", headers=headers, json=payload)
        res.raise_for_status()
        data = res.json().get("results", [])
        return data

@mcp.tool()
async def notion_retrieve_page(page_id: str, user_id: str) -> Dict[str, Any]:
    """Retrieve a Notion page by its ID."""
    print(f"📄 RETRIEVE_PAGE: user={user_id} page={page_id}")
    headers = await get_notion_headers(user_id)
    clean_id = page_id.replace("-", "")
    async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
        res = await client.get(f"https://api.notion.com/v1/pages/{clean_id}", headers=headers)
        res.raise_for_status()
        return res.json()

@mcp.tool()
async def notion_query_database(database_id: str, user_id: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Query a Notion database with optional complex filters/sorts."""
    print(f"🔎 QUERY_DATABASE: user={user_id} db={database_id}")
    headers = await get_notion_headers(user_id)
    clean_id = database_id.replace("-", "")
    payload = {}
    if filter_dict:
        payload["filter"] = filter_dict
    async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
        res = await client.post(f"https://api.notion.com/v1/databases/{clean_id}/query", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()

@mcp.tool()
async def notion_retrieve_database(database_id: str, user_id: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific database ID."""
    print(f"📂 RETRIEVE_DATABASE: user={user_id} db={database_id}")
    headers = await get_notion_headers(user_id)
    clean_id = database_id.replace("-", "")
    async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
        res = await client.get(f"https://api.notion.com/v1/databases/{clean_id}", headers=headers)
        res.raise_for_status()
        return res.json()

@mcp.tool()
def notion_resolve_link(url: str) -> str:
    """Extract UUID from link."""
    print(f"🔗 RESOLVE_LINK: {url}")
    match = re.search(r"([a-f0-9]{32})", url.replace("-", ""))
    res = match.group(1) if match else "invalid"
    print(f"   -> Result: {res}")
    return res

@mcp.tool()
async def notion_inspect_id(id_string: str, user_id: str) -> Dict[str, Any]:
    """Retrieve metadata about a Notion ID to determine if it is a page or a database. Accepts IDs or URLs."""
    print(f"🧐 INSPECT_ID: user={user_id} input='{id_string}'")
    headers = await get_notion_headers(user_id)
    
    # Resolve if URL or already cleaned ID
    id_string = str(id_string or "")
    match = re.search(r"([a-f0-9]{32})", id_string.replace("-", ""))
    clean_id = match.group(1) if match else id_string.replace("-", "")
    
    if not clean_id:
        return {"type": "unknown", "id": "none", "error": "Empty ID provided"}

    print(f"   -> Resolved ID: {clean_id}")
    
    db_status = None
    page_status = None
    db_error = None
    page_error = None

    try:
        async with httpx.AsyncClient(timeout=NOTION_TIMEOUT) as client:
            # Try database first via retrieve endpoint
            try:
                res_db = await client.get(f"https://api.notion.com/v1/databases/{clean_id}", headers=headers)
                db_status = res_db.status_code
                if res_db.status_code == 200:
                    data = res_db.json()
                    print(f"✅ Found DATABASE: {data.get('title', [{}])[0].get('plain_text', 'Untitled')}")
                    return {"type": "database", "id": clean_id, "metadata": data}
                db_error = res_db.text
            except httpx.TimeoutException:
                print(f"⚠️  Database check timed out for {clean_id}")
                
            # Try page via retrieve endpoint
            try:
                res_page = await client.get(f"https://api.notion.com/v1/pages/{clean_id}", headers=headers)
                page_status = res_page.status_code
                if res_page.status_code == 200:
                    data = res_page.json()
                    print(f"✅ Found PAGE")
                    return {"type": "page", "id": clean_id, "metadata": data}
                page_error = res_page.text
            except httpx.TimeoutException:
                print(f"⚠️  Page check timed out for {clean_id}")

        error_msg = (
            f"ID not found or inaccessible. "
            f"database_status={db_status}, page_status={page_status}"
        )
        print(f"❌ {error_msg}: {clean_id}")
        return {
            "type": "unknown",
            "id": clean_id,
            "error": error_msg,
            "details": {
                "database_status": db_status,
                "page_status": page_status,
                "database_error": db_error,
                "page_error": page_error,
            },
        }
    except Exception as e:
        print(f"❌ Error in inspect_id: {str(e)}")
        return {"type": "unknown", "id": clean_id, "error": str(e)}

@mcp.tool()
async def query_agent_memory(user_id: str, category: str) -> List[Dict[str, Any]]:
    """
    List the short-term memory pointers (snippets) for a specific user and category.
    Categories: casual | retrieval | meeting | publishing
    Use this to see what the agent has previously discussed or generated.
    """
    print(f"🧠 QUERY_MEMORY: user={user_id} cat={category}")
    key = f"agent_index:{user_id}:{category}"
    raw_data = redis_client.hgetall(key)
    if not raw_data:
        return []
    
    results = []
    for k, v in raw_data.items():
        try:
            results.append(json.loads(v))
        except:
            results.append(v)
            
    # Sort by timestamp descending
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results

@mcp.tool()
async def get_archived_conversation(msg_id: str) -> Dict[str, Any]:
    """
    Retrieve full conversation details from the persistent archive.
    Use the 'msg_id' found from query_agent_memory to fetch the full payload.
    """
    print(f"📦 RETRIEVE_ARCHIVE: id='{msg_id}'")
    key = f"conv_archive:{msg_id}"
    data = redis_client.get(key)
    if not data:
        return {"error": "Conversation not found in archive."}
    
    try:
        if isinstance(data, (str, bytes)):
            return json.loads(data)
        return data
    except Exception as e:
        return {"error": f"Parse error: {str(e)}", "raw": str(data)}

if __name__ == "__main__":
    print("🚀 NOTION MCP SERVER STARTED (SSE MODE)")
    print("📡 Listening on port 8000...")
    mcp.run(transport="sse")
