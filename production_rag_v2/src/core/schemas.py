from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MeetingNote(BaseModel):
    """Invariant 3: Schema for formatted meeting notes."""
    name: str = Field(..., description="The name/title of the meeting session")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="Date of the meeting")
    summary: str = Field(..., description="The summarized content and action items")

class QAPair(BaseModel):
    """Invariant 3: Schema for atomic Q&A exports."""
    question: str = Field(..., description="The user's query")
    answer: str = Field(..., description="The system's response")
    added_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Time of export")

class AgenticPointer(BaseModel):
    """The 'Short-Term Memory' index entries stored in AgentState."""
    msg_id: str = Field(..., description="The unique ID to fetch full data from Redis")
    category: str = Field(..., description="casual | retrieval | meeting | publishing")
    snippet: str = Field(..., description="1-2 sentence preview for agentic 'glance'")
    status: str = Field("final", description="draft | final | error")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    ref_id: Optional[str] = Field(None, description="External platform reference (Notion Page/DB ID)")

class ConversationArchive(BaseModel):
    """The 'Full-Payload' stored in the Redis conv_archive."""
    msg_id: str
    category: str
    query: str = Field(..., description="The original user query")
    payload: Dict[str, Any] = Field(..., description="Raw content blobs (answers, summaries, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sources, latencies, versions")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
