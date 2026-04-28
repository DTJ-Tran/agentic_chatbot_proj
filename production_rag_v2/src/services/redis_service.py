from upstash_redis import Redis
from src.core.config import settings

class RedisService:
    """
    Singleton service to manage Redis connections and avoid session leaks.
    """
    _instance = None
    _client = None

    @classmethod
    def get_client(cls) -> Redis:
        if cls._client is None:
            cls._client = Redis(url=settings.redis_url, token=settings.redis_token)
        return cls._client

    @classmethod
    def cache_conversation(cls, msg_id: str, data: dict):
        """Store full conversation data in Redis with undefined TTL."""
        client = cls.get_client()
        key = f"conv_archive:{msg_id}"
        import json
        client.set(key, json.dumps(data))

    @classmethod
    def get_cached_conversation(cls, msg_id: str) -> dict:
        """Retrieve full conversation data from Redis."""
        client = cls.get_client()
        key = f"conv_archive:{msg_id}"
        data = client.get(key)
        import json
        return json.loads(data) if data else None

    @classmethod
    def update_index(cls, user_id: str, category: str, pointer: dict):
        """Update the agent's short-term memory index in Redis."""
        client = cls.get_client()
        key = f"agent_index:{user_id}:{category}"
        import json
        # We store as a hash (HSET) where field is msg_id
        client.hset(key, pointer["msg_id"], json.dumps(pointer))

    @classmethod
    def query_index(cls, user_id: str, category: str) -> list:
        """Query the agent's short-term memory index from Redis."""
        client = cls.get_client()
        key = f"agent_index:{user_id}:{category}"
        import json
        all_pointers = client.hgetall(key)
        if not all_pointers:
            return []
        return [json.loads(v) for v in all_pointers.values()]

    @classmethod
    def cleanup(cls):
        """Clean up the Redis client."""
        if cls._client:
            print("🧹 [RedisService] Closing connections...")
            # Upstash Redis client doesn't have an explicit close() 
            # but we clear the reference and trigger GC to ensure 
            # underlying transport handles are released.
            cls._client = None
            import gc
            gc.collect()

redis_service = RedisService()
