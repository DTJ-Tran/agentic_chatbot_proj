import motor.motor_asyncio
from typing import Dict, Any, List, Optional
from src.core.config import settings

class MongoDBService:
    """
    Handles persistence of meeting notes and ASR transcripts in MongoDB.
    """
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongo_url)
        self.db = self.client[settings.mongo_db_name]
        self.meetings_col = self.db["meeting_notes"]
        self.transcripts_col = self.db["transcripts"]

    async def save_transcript_segment(self, session_id: str, segment: Dict[str, Any]):
        """Saves a single transcribed segment."""
        segment["session_id"] = session_id
        await self.transcripts_col.insert_one(segment)

    async def save_meeting_note(self, session_id: str, note_data: Dict[str, Any]):
        """Saves the final summarized meeting note."""
        note_data["session_id"] = session_id
        await self.meetings_col.update_one(
            {"session_id": session_id},
            {"$set": note_data},
            upsert=True
        )

    async def get_meeting_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves history for a meeting session."""
        cursor = self.transcripts_col.find({"session_id": session_id}).sort("absolute_start_time", 1)
        return await cursor.to_list(length=1000)

    async def close(self):
        self.client.close()

# Singleton instance
mongodb_service = MongoDBService()
