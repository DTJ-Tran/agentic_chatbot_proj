import os
import httpx
import asyncio
import numpy as np
import soundfile as sf
import io
import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.config import settings
from src.services.identity_service import get_identity_service
from src.services.vad_service import get_vad_service

class DiarizationService:
    """
    Speaker Diarization service using pyannote.ai managed REST API.
    Handles buffering to optimize API overhead and provides consistent speaker tracking.
    """
    def __init__(self):
        self.api_key = settings.pyannot_api
        if not self.api_key:
            print("⚠️ Warning: PYANNOT_API key not found in settings")
        self.base_url = "https://api.pyannote.ai/v1"
        
        # Buffering state
        self._audio_buffer = []
        self._buffer_duration = 0.0
        self._buffer_threshold = 10.0  # seconds
        self._overlap_duration = 3.0   # seconds to overlap between windows
        
        # Global timeline tracking
        self._stream_time = 0.0         # Total time processed so far
        self._buffer_start_time = 0.0   # Where the current buffer starts in global time
        self._last_reported_time = 0.0  # Last time we emitted finalized segments
        
        # Last known state from API
        self._last_segments = []        # Mapped segments
        self._last_speaker = "SPEAKER_00"
        self._to_report = []            # Buffer of finalized segments for the caller
        self._is_processing = False
        
        # Worker Queue (Init but don't start task yet)
        self._job_queue = asyncio.Queue()
        self._worker_task = None
        
        # Persistent client
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=30.0
        )

    async def _job_worker(self):
        """Background worker that processes diarization jobs sequentially from the queue."""
        while True:
            audio_data, sr, buffer_offset = await self._job_queue.get()
            try:
                self._is_processing = True
                await self._run_handshake(audio_data, sr, buffer_offset)
            except Exception as e:
                print(f"❌ Diarization worker error: {e}")
            finally:
                self._is_processing = False
                self._job_queue.task_done()

    async def _get_presigned_url(self, object_key: str) -> str:
        # ... (rest of the helper methods remain same)
        url = f"{self.base_url}/media/input"
        resp = await self.client.post(url, json={"url": f"media://{object_key}"})
        resp.raise_for_status()
        return resp.json()["url"]

    async def _upload_audio(self, presigned_url: str, binary_data: bytes):
        async with httpx.AsyncClient() as upload_client:
            resp = await upload_client.put(
                presigned_url, 
                content=binary_data
            )
            resp.raise_for_status()

    async def _trigger_diarization(self, object_key: str) -> str:
        url = f"{self.base_url}/diarize"
        resp = await self.client.post(url, json={"url": f"media://{object_key}"})
        resp.raise_for_status()
        return resp.json()["jobId"]

    def _calculate_stitching_map(self, raw_segments: List[Dict[str, Any]], buffer_offset: float) -> Dict[str, str]:
        """
        Voting-based algorithm to map local IDs from the current job to persistent global IDs.
        Analyzes the overlap window (the shared part between current job and previous known state).
        """
        overlap_limit = self._overlap_duration
        overlap_start_global = buffer_offset
        overlap_end_global = buffer_offset + overlap_limit
        
        prev_overlap = [
            s for s in self._last_segments 
            if s["end"] > overlap_start_global and s["start"] < overlap_end_global
        ]
        
        curr_overlap = [
            s for s in raw_segments
            if s["start"] < overlap_limit
        ]
        
        if not prev_overlap:
            # Map any found speakers to initial GLOBAL IDs
            return {s["speaker"]: s["speaker"].replace("SPEAKER", "GLOBAL") for s in raw_segments}

        mapping = {}
        local_speakers = set(s["speaker"] for s in raw_segments)
        
        for local_spk in local_speakers:
            local_spk_segments = [s for s in curr_overlap if s["speaker"] == local_spk]
            votes = {}
            
            for l_seg in local_spk_segments:
                l_start = max(0, l_seg["start"])
                l_end = min(overlap_limit, l_seg["end"])
                
                for p_seg in prev_overlap:
                    p_start_rel = max(0, p_seg["start"] - buffer_offset)
                    p_end_rel = min(overlap_limit, p_seg["end"] - buffer_offset)
                    inter_start = max(l_start, p_start_rel)
                    inter_end = min(l_end, p_end_rel)
                    
                    if inter_end > inter_start:
                        duration = inter_end - inter_start
                        votes[p_seg["speaker"]] = votes.get(p_seg["speaker"], 0) + duration
            
            if votes:
                best_global = max(votes.items(), key=lambda x: x[1])[0]
                mapping[local_spk] = best_global
            else:
                mapping[local_spk] = local_spk.replace("SPEAKER", f"GLOBAL_NEW_{uuid.uuid4().hex[:4]}")

        return mapping

    async def _poll_job(self, job_id: str, buffer_offset: float, audio_data: np.ndarray, sr: int):
        id_service = get_identity_service()
        url = f"{self.base_url}/jobs/{job_id}"
        while True:
            resp = await self.client.get(url)
            resp.raise_for_status()
            status_data = resp.json()
            
            if status_data["status"] == "succeeded":
                output_data = status_data.get("output", {})
                raw_segments = output_data.get("diarization", [])
                print(f"\nDEBUG: Job {job_id} raw segments: {len(raw_segments)}")
                # for s in raw_segments[:3]: print(f"  {s}") # Print first few if needed
                
                mapping = self._calculate_stitching_map(raw_segments, buffer_offset)
                
                # Embedding-based Persistent Identity Re-mapping
                # Group raw segments by local speaker ID to find best clips for identity matching
                speaker_clips: Dict[str, List[np.ndarray]] = {}
                for seg in raw_segments:
                    spk = seg["speaker"]
                    # Extract 1-3 seconds of audio for embedding if possible
                    duration = seg["end"] - seg["start"]
                    if duration < 0.5: continue # Too short
                    
                    s_idx = int(seg["start"] * sr)
                    e_idx = int(seg["end"] * sr)
                    clip = audio_data[s_idx:e_idx]
                    
                    if spk not in speaker_clips: speaker_clips[spk] = []
                    speaker_clips[spk].append(clip)

                # Final ID mapping: Local Speaker -> Persistent Speaker
                persistent_mapping = {}
                for local_spk in set(seg["speaker"] for seg in raw_segments):
                    clips = speaker_clips.get(local_spk, [])
                    if clips:
                        # Use the longest clip for best embedding
                        best_clip = max(clips, key=len)
                        persistent_id = await id_service.match_speaker(best_clip, sr)
                        persistent_mapping[local_spk] = persistent_id
                    else:
                        # Fallback to local ID if no good clips found (unlikely)
                        persistent_mapping[local_spk] = local_spk

                new_mapped_segments = []
                for seg in raw_segments:
                    # We prioritize the persistent identity from embedding registry
                    spk = persistent_mapping.get(seg["speaker"], seg["speaker"])
                    
                    new_mapped_segments.append({
                        "start": seg["start"] + buffer_offset,
                        "end": seg["end"] + buffer_offset,
                        "speaker": spk
                    })
                
                # Update history for next stitch
                self._last_segments = new_mapped_segments
                if self._last_segments:
                    self._last_speaker = self._last_segments[-1]["speaker"]
                    
                    # Push finalized segments to report buffer
                    for seg in new_mapped_segments:
                        if seg["end"] <= self._last_reported_time:
                            continue
                        
                        # Adjust start if it overlaps with what we already reported
                        report_seg = seg.copy()
                        if report_seg["start"] < self._last_reported_time:
                            report_seg["start"] = self._last_reported_time
                        
                        self._to_report.append(report_seg)
                        self._last_reported_time = report_seg["end"]
                
                print(f"✅ Diarization Job {job_id} stitched. Map: {mapping} | Segments: {len(raw_segments)}")
                break
            elif status_data["status"] == "failed":
                print(f"⚠️ Diarization job {job_id} failed: {status_data}")
                break
            await asyncio.sleep(0.5)

    async def _run_handshake(self, audio_data: np.ndarray, sr: int, buffer_offset: float):
        object_key = f"chunk_{uuid.uuid4().hex[:8]}"
        try:
            # Convert to int16 PCM for robust API compatibility
            audio_int16 = (audio_data * 32767).astype(np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, audio_int16, sr, format='WAV', subtype='PCM_16')
            binary_data = buffer.getvalue()
            presigned_url = await self._get_presigned_url(object_key)
            await self._upload_audio(presigned_url, binary_data)
            job_id = await self._trigger_diarization(object_key)
            await self._poll_job(job_id, buffer_offset, audio_data, sr)
        except Exception as e:
            print(f"❌ Diarization handshake error for offset {buffer_offset}: {e}")

    async def process(self, audio_data: np.ndarray, sr: int = 16000) -> List[Dict[str, Any]]:
        """Process incoming audio chunks using a sliding window."""
        # Lazy start worker task
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._job_worker())
            
        duration = len(audio_data) / sr
        start_time = self._stream_time
        
        self._audio_buffer.append(audio_data)
        self._buffer_duration += duration
        
        if len(self._audio_buffer) == 1:
            self._buffer_start_time = start_time
            
        if self._buffer_duration >= self._buffer_threshold:
            full_audio = np.concatenate(self._audio_buffer)
            offset_for_job = self._buffer_start_time
            
            overlap_samples = int(self._overlap_duration * sr)
            if len(full_audio) > overlap_samples:
                self._audio_buffer = [full_audio[-overlap_samples:]]
                self._buffer_duration = self._overlap_duration
                self._buffer_start_time = (start_time + duration) - self._overlap_duration
            else:
                self._audio_buffer = []
                self._buffer_duration = 0.0
            
            # VAD Check: Skip Pyannote API if no significant speech detected
            vad = get_vad_service()
            if vad.is_speech_present(full_audio, sr):
                await self._job_queue.put((full_audio, sr, offset_for_job))
            else:
                print(f"🔇 VAD: Skipping silent segment at {offset_for_job:.2f}s")
                # Add silence segment to reporting queue to maintain timeline
                self._to_report.append({
                    "start": offset_for_job,
                    "end": offset_for_job + self._buffer_threshold,
                    "speaker": "SILENCE"
                })
                self._last_reported_time = offset_for_job + self._buffer_threshold
            
        self._stream_time += duration
        
        # Return and clear the report buffer
        if self._to_report:
            report = list(self._to_report)
            self._to_report.clear()
            return report
            
        return []

# Singleton
diarization_service = DiarizationService()
