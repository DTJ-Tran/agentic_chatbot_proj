import numpy as np
import librosa
from typing import Union, Tuple, Dict, Any
from pathlib import Path

class AcousticService:
    """
    Acoustic analysis service for Signal-to-Noise Ratio (SNR) and Voice Activity Detection (VAD).
    Acts as the primary gate for ASR routing.
    """
    def __init__(self, silence_threshold: float = 0.01):
        self.silence_threshold = silence_threshold

    def calculate_snr(self, audio: np.ndarray) -> float:
        """
        Estimates the Signal-to-Noise Ratio (SNR) in dB.
        Uses a simple energy-based estimation: Signal is high energy parts, Noise is low energy.
        """
        if len(audio) == 0:
            return 0.0
        
        # Calculate frame energies
        win_length = 512
        hop_length = 256
        frames = librosa.util.frame(audio, frame_length=win_length, hop_length=hop_length)
        energies = np.sum(frames**2, axis=0) / win_length
        
        if len(energies) == 0:
            return 0.0
            
        # Distinguish signal and noise using threshold
        # We take the top 30% as signal and bottom 10% as noise estimate
        sorted_energies = np.sort(energies)
        n_frames = len(sorted_energies)
        
        noise_part = sorted_energies[:max(1, int(0.1 * n_frames))]
        signal_part = sorted_energies[int(0.7 * n_frames):]
        
        noise_power = np.mean(noise_part) if len(noise_part) > 0 else 1e-10
        signal_power = np.mean(signal_part) if len(signal_part) > 0 else 1e-10
        
        if noise_power < 1e-15: # Prevent log of zero
            noise_power = 1e-15
            
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def is_speech(self, audio: np.ndarray, sr: int = 16000) -> bool:
        """
        Basic energy-based VAD.
        Returns True if the max amplitude exceeds the silence threshold.
        """
        if len(audio) == 0:
            return False
            
        max_val = np.abs(audio).max()
        return max_val > self.silence_threshold

    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """Compares SNR and VAD in one pass."""
        return {
            "is_speech": self.is_speech(audio),
            "snr_db": self.calculate_snr(audio)
        }

# Singleton
acoustic_service = AcousticService()
