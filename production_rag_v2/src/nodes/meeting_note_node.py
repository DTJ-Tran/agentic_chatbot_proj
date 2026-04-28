import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, Any
from src.services.asr_pipeline import asr_pipeline
from src.services.mongodb_service import mongodb_service
from src.services.graphdb_service import graphdb_service
from src.services.llm_service import llm_service
from src.services.nlp_service import nlp_service
from src.engine.state import AgentState
from src.core.schemas import AgenticPointer, ConversationArchive
from src.services.redis_service import RedisService

class MeetingNoteNode:
    """
    LangGraph node for coordinating the ASR meeting workstream.
    Ensures data consistency with AgenticPointer and ConversationArchive.
    """
    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        msg = state.get("msg", {})
        msg_body = msg.get("msg_body", {})
        intent = msg_body.get("meeting_intent")
        session_id = state.get("session_id", str(uuid.uuid4()))
        response_content = ""

        if intent == "start_meeting":
            asr_pipeline.start_meeting(session_id)
            response_content = f"Meeting session {session_id} started. I am now listening."
            graphdb_service.create_meeting_node(
                session_id=session_id,
                title=msg_body.get("title", "Untitled Meeting"),
                date=msg_body.get("date", datetime.now().strftime("%Y-%m-%d"))
            )

        elif intent == "process_audio":
            audio_path = msg_body.get("audio_path")
            if audio_path:
                await asr_pipeline.add_segment(session_id, audio_path)
                session_state = asr_pipeline.get_session_state(session_id)
                if session_state.get("transcripts"):
                    latest_seg = session_state["transcripts"][-1]
                    await mongodb_service.save_transcript_segment(session_id, latest_seg)
                    
                    transcript_text = latest_seg.get("text", "")
                    if transcript_text:
                        async def _bg_ingestion(text: str, sid: str):
                            try:
                                enriched = await nlp_service.extract_enriched_metadata(text)
                                for person in enriched.get("persons", []):
                                    graphdb_service.link_meeting_participant_ephemeral(sid, person)
                                for concept in enriched.get("concepts", []):
                                    graphdb_service.link_concept(sid, concept["name"], concept["type"])
                            except Exception: pass
                        asyncio.create_task(_bg_ingestion(transcript_text, session_id))
                            
                response_content = "Processing audio segment and updating Knowledge Graph..."
            else:
                response_content = "Error: No audio path provided."

        elif intent in ["end_meeting", "summarize_meeting"]:
            session_data = asr_pipeline.finalize_meeting(session_id)
            if "error" in session_data:
                response_content = f"Error ending meeting: {session_data['error']}"
            else:
                transcript = session_data.get("full_transcript", "")
                summary_prompt = f"Summarize the following meeting transcript into key action items:\n\n{transcript}"
                summary = await llm_service.generate(summary_prompt)
                
                # Persist to Mongo
                await mongodb_service.save_meeting_note(session_id, {
                    "transcript": transcript,
                    "summary": summary,
                    "status": "completed",
                    "metadata": session_data.get("metadata", {})
                })
                response_content = f"Meeting summarized.\n\nSummary:\n{summary}"

        # --- DATA CONSISTENCY BLOCK ---
        indexing_update = {}
        generated_summary = locals().get("summary")
        
        if generated_summary:
            # 1. Create Harmonized Pointer
            pointer = AgenticPointer(
                msg_id=session_id,
                category="meeting",
                snippet=generated_summary[:150] + "...",
                status="final"
            )
            
            # 2. Create Harmonized Archive
            archive = ConversationArchive(
                msg_id=session_id,
                category="meeting",
                query=msg_body.get("title", f"Meeting {session_id}"),
                payload={"summary": generated_summary},
                metadata={"transcript_len": len(locals().get("transcript", ""))}
            )
            
            # 3. State & Redis Persistence
            indexing_update["meeting_conv"] = {session_id: pointer.model_dump()}
            RedisService.cache_conversation(session_id, archive.model_dump())
            
            user_id = msg.get("user_id", "default_user")
            RedisService.update_index(user_id, "meeting", pointer.model_dump())

        # Update msg for UI
        new_msg = msg.copy()
        new_msg["msg_content"] = response_content

        return {
            "msg": new_msg, 
            "summary": generated_summary,
            "route": "meeting",
            **indexing_update
        }

meeting_node = MeetingNoteNode()
