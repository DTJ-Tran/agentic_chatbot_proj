import onnxruntime as ort
import numpy as np
import os
from typing import List, Dict, Any, Optional

class VADService:
    """
    Voice Activity Detection service using Silero VAD ONNX model.
    Filters out silent segments to optimize downstream API costs and noise.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VAD model not found at {model_path}")
        
        # Initialize ONNX session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Model properties
        self.sample_rates = [8000, 16000]
        self._window_size_samples = 512 # Default for 16kHz
        
    def _reset_state(self, batch_size: int = 1):
        """Initial state for Silero VAD."""
        return np.zeros((2, batch_size, 128)).astype(np.float32)

    def is_speech_present(self, audio: np.ndarray, sr: int = 16000, threshold: float = 0.5, min_speech_duration_ms: int = 250) -> bool:
        """
        Scan an audio segment and return True if significant speech is detected.
        
        Args:
            audio: np.ndarray of float32 samples.
            sr: Sample rate (must be 8000 or 16000).
            threshold: Probability threshold for speech.
            min_speech_duration_ms: Minimum cumulative speech duration in ms to return True.
        """
        if sr not in self.sample_rates:
            raise ValueError(f"SR {sr} not supported by Silero VAD. Use 8000 or 16000.")

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        window_size = 512 if sr == 16000 else 256
        state = self._reset_state()
        sr_tensor = np.array([sr], dtype=np.int64)
        
        speech_samples = 0
        total_samples = len(audio)
        
        for i in range(0, total_samples - window_size, window_size):
            chunk = audio[i:i + window_size]
            if len(chunk) < window_size:
                break
                
            ort_inputs = {
                'input': chunk.reshape(1, -1),
                'sr': sr_tensor,
                'state': state
            }
            
            ort_outs = self.session.run(None, ort_inputs)
            out = ort_outs[0]
            state = ort_outs[1]
            
            prob = out[0][0]
            if prob >= threshold:
                speech_samples += window_size
                
        speech_duration_ms = (speech_samples / sr) * 1000
        # print(f"DEBUG: VAD speech_duration_ms={speech_duration_ms:.2f}ms")
        return speech_duration_ms >= min_speech_duration_ms

    def get_speech_timestamps(self, audio: np.ndarray, sr: int = 16000, 
                             threshold: float = 0.5, 
                             min_speech_duration_ms: int = 250,
                             min_silence_duration_ms: int = 100) -> List[Dict[str, float]]:
        """
        Detect speech segments in an audio array.
        Returns a list of dictionaries with 'start' and 'end' in seconds.
        """
        if sr not in self.sample_rates:
            raise ValueError(f"SR {sr} not supported. Use 8000 or 16000.")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        window_size = 512 if sr == 16000 else 256
        state = self._reset_state()
        sr_tensor = np.array([sr], dtype=np.int64)
        
        speech_segments = []
        is_speech = False
        current_start = 0
        
        total_samples = len(audio)
        
        for i in range(0, total_samples - window_size, window_size):
            chunk = audio[i:i + window_size]
            if len(chunk) < window_size:
                break
                
            ort_inputs = {
                'input': chunk.reshape(1, -1),
                'sr': sr_tensor,
                'state': state
            }
            
            ort_outs = self.session.run(None, ort_inputs)
            out = ort_outs[0]
            state = ort_outs[1]
            
            prob = out[0][0]
            
            if prob >= threshold and not is_speech:
                is_speech = True
                current_start = i
            elif prob < threshold and is_speech:
                is_speech = False
                # End of segment
                segment_duration_ms = ((i - current_start) / sr) * 1000
                if segment_duration_ms >= min_speech_duration_ms:
                    speech_segments.append({
                        'start': float(current_start / sr),
                        'end': float(i / sr)
                    })
        
        # Handle trailing segment
        if is_speech:
            segment_duration_ms = ((total_samples - current_start) / sr) * 1000
            if segment_duration_ms >= min_speech_duration_ms:
                speech_segments.append({
                    'start': float(current_start / sr),
                    'end': float(total_samples / sr)
                })
                
        # Merge segments that are close to each other (silence < min_silence_duration_ms)
        if not speech_segments:
            return []
            
        merged_segments = []
        header = speech_segments[0].copy()
        
        for i in range(1, len(speech_segments)):
            curr = speech_segments[i]
            silence_duration_ms = (curr['start'] - header['end']) * 1000
            if silence_duration_ms < min_silence_duration_ms:
                header['end'] = curr['end']
            else:
                merged_segments.append(header)
                header = curr.copy()
        merged_segments.append(header)
        
        return merged_segments

from src.core.config import settings

# Helper to get singleton
_vad_service = None

def get_vad_service():
    global _vad_service
    if _vad_service is None:
        _vad_service = VADService(settings.vad_model_path)
    return _vad_service
