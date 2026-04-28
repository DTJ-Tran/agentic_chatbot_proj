## THIS CODE IS RUN ON DIFFERENCE VM - no need to run with in this project
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
import httpx
import os
import secrets
import json
from urllib.parse import urlencode
from dotenv import load_dotenv
from upstash_redis import Redis

# Load env
load_dotenv()

app = FastAPI()

# Upstash Redis client
redis_client = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
)

# =========================
# 🔐 OAuth Connect
# =========================
@app.get("/api/notion/oauth/connect")
async def connect_notion(user_id: str):
    state = secrets.token_urlsafe(32)

    # store state → user_id (TTL 5 minutes)
    redis_client.set(f"oauth_state:{state}", user_id, ex=300)

    params = {
        "client_id": os.getenv("NOTION_CLIENT_ID"),
        "response_type": "code",
        "owner": "user",
        "redirect_uri": os.getenv("NOTION_REDIRECT_URI"),
        "state": state
    }

    url = "https://api.notion.com/v1/oauth/authorize?" + urlencode(params)
    return RedirectResponse(url)


# =========================
# 🔄 Exchange Code → Token
# =========================
async def exchange_code_for_token(code: str):
    url = "https://api.notion.com/v1/oauth/token"

    auth = (
        os.getenv("NOTION_CLIENT_ID"),
        os.getenv("NOTION_CLIENT_SECRET")
    )

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": os.getenv("NOTION_REDIRECT_URI"),
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(url, data=data, auth=auth)

    if res.status_code != 200:
        raise Exception(f"OAuth failed: {res.text}")

    return res.json()


# =========================
# 👤 Get Notion Identity
# =========================
async def get_notion_user(access_token: str):
    url = "https://api.notion.com/v1/users/me"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Notion-Version": "2022-06-28"
    }

    async with httpx.AsyncClient() as client:
        res = await client.get(url, headers=headers)

    return res.json()


# =========================
# 🔁 OAuth Callback
# =========================
@app.get("/api/notion/oauth/callback")
async def notion_callback(
    code: str = Query(None),
    state: str = Query(None)
):
    if not code or not state:
        return {"error": "Missing OAuth parameters"}

    # validate state
    user_id = redis_client.get(f"oauth_state:{state}")
    if not user_id:
        return {"error": "Invalid or expired state"}

    # cleanup
    redis_client.delete(f"oauth_state:{state}")

    # exchange token
    token_data = await exchange_code_for_token(code)
    access_token = token_data["access_token"]

    # fetch Notion identity
    notion_user = await get_notion_user(access_token)

    bot_data = notion_user.get("bot", {})


    # store token (persistent)
    redis_client.set(
        f"user_token:{user_id}",
        json.dumps({
            "access_token": access_token,
            "workspace_id": token_data.get("workspace_id"),
            "notion_user_id": notion_user.get("id"),
            "workspace_name": bot_data.get("workspace_name") # Access from the nested 'bot' key``
        }),
        ex=3600 # store within 1 hours
    )

    return {
        "message": "Connected",
        "user_id": user_id,
        "workspace": notion_user.get("workspace_name")
    }


# =========================
# 🔍 Get Stored Token
# =========================
def get_user_token(user_id: str):
    data = redis_client.get(f"user_token:{user_id}")
    if not data:
        return None
    return json.loads(data)