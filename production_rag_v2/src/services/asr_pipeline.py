import asyncio
import numpy as np
import time
from typing import List, Dict, Any, Optional
from src.services.asr_service import asr_service
from src.services.fallback_asr_service import fallback_asr_service
from src.services.acoustic_service import acoustic_service
from src.services.kenlm_service import kenlm_service
from src.services.diarization_service import diarization_service
from src.core.config import settings



class ASRPipeline:
    """
    Advanced ASR Pipeline with Gated Acoustic Routing and KenLM Rescoring.
    """
    def __init__(self):
        self.asr = asr_service
        self.fallback = fallback_asr_service
        self.acoustic = acoustic_service
        self.diarizer = diarization_service
        self.rescorer = kenlm_service

    async def process_segment(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single audio segment through the advanced pipeline.
        """
        start_time = time.time()
        
        # 1. Acoustic VAD & SNR Filter
        vad_info = self.acoustic.analyze(audio_data)
        if not vad_info["is_speech"]:
            return {"text": "", "status": "silent", "confidence": 0.0}

        # 2. Parallel Diarization (starts early)
        diarization_task = asyncio.create_task(self.diarizer.process(audio_data))

        # 3. Gated ASR Routing
        result = None
        routing_reason = "Base"
        
        if vad_info["snr_db"] < settings.snr_threshold_db:
            # High noise -> Immediate Fallback
            result = self.fallback.transcribe(audio_data)
            routing_reason = "Acoustic-Fallback"
        else:
            # Good signal -> Try Base Moonshine
            result = self.asr.transcribe(audio_data)
            
            # 4. Confidence-based Late Routing
            if result.get("confidence", 1.0) < settings.asr_confidence_threshold:
                print(f"⚠️ Low confidence ({result['confidence']:.2f}) from Moonshine. Triggering Fallback...")
                fallback_result = self.fallback.transcribe(audio_data)
                
                # Rescore hypotheses if multiple are available (simplified here)
                result = fallback_result
                routing_reason = "Confidence-Fallback"

        # 5. Wait for Diarization
        diarization = await diarization_task
        
        # 6. Alignment (MFA Removed - using placeholder)
        final_segments = [{
            "word": result["text"],
            "start": 0.0,
            "end": float(len(audio_data)) / 16000.0,
            "speaker": diarization[0]["speaker"] if diarization else "UNKNOWN"
        }]
        
        # 7. Rescoring / Confidence Logging
        # Use bilingual rescorer to detect language and normalize score
        lm_info = self.rescorer.get_bilingual_score(result["text"])
        print(f"DEBUG: Transcription {lm_info['lang']} (Score: {lm_info['score']:.2f}) | Engine: {result.get('engine', 'Unknown')}")
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "text": result["text"],
            "segments": final_segments,
            "confidence": result.get("confidence", 0.0),
            "lm_score": lm_info["score"],
            "language": lm_info["lang"],
            "kenlm_confidence": lm_info["confidence"],
            "latency_ms": latency_ms,
            "routing": routing_reason,
            "snr_db": vad_info["snr_db"],
            "engine": result.get("engine", "Unknown")
        }

    def start_meeting(self, session_id: str) -> None:
        if not hasattr(self, 'active_sessions'):
            self.active_sessions = {}
        self.active_sessions[session_id] = {"transcripts": [], "raw_audio_paths": [], "current_offset_sec": 0.0}
        print(f"🎙️ [ASR Pipeline] Started real-time session: {session_id}")

    async def add_segment(self, session_id: str, audio_path: str) -> Dict[str, Any]:
        print(f"🎙️ [ASR Pipeline] Processing segment for {session_id}: {audio_path}")
        if not hasattr(self, 'active_sessions'):
            self.active_sessions = {}
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {"transcripts": [], "raw_audio_paths": [], "current_offset_sec": 0.0}
            
        import librosa
        try:
            audio_data, sr = librosa.load(audio_path, sr=16000)
            result = await self.process_segment(audio_data)
            
            # Enforce absolutely deterministic temporal chaining
            offset = self.active_sessions[session_id].get("current_offset_sec", 0.0)
            for seg in result.get("segments", []):
                seg["absolute_start_time"] = offset + seg.get("start", 0.0)
                seg["absolute_end_time"] = offset + seg.get("end", 0.0)
            result["absolute_start_time"] = offset
            
            duration = float(len(audio_data)) / sr
            self.active_sessions[session_id]["current_offset_sec"] = offset + duration
            
            self.active_sessions[session_id]["transcripts"].append(result)
            self.active_sessions[session_id]["raw_audio_paths"].append(audio_path)
            return result
        except Exception as e:
            print(f"⚠️ [ASR Pipeline] Error reading audio {audio_path}: {e}")
            return {"error": str(e)}

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        if not hasattr(self, 'active_sessions'):
            return {}
        return self.active_sessions.get(session_id, {})

    def finalize_meeting(self, session_id: str) -> Dict[str, Any]:
        print(f"🎙️ [ASR Pipeline] Finalizing session: {session_id}")
        if not hasattr(self, 'active_sessions') or session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session_data = self.active_sessions[session_id]
        
        full_transcript = "\n".join([t.get("text", "") for t in session_data["transcripts"] if t.get("text")])
        
        del self.active_sessions[session_id]
        
        return {
            "full_transcript": full_transcript,
            "metadata": {
                "segment_count": len(session_data["transcripts"])
            }
        }

# Singleton instance
asr_pipeline = ASRPipeline()
