import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import onnxruntime as ort
from transformers import Wav2Vec2Processor
import librosa
from src.core.config import settings
from src.services.phoneme_service import phoneme_service
from src.services.beam_search_service import BeamSearchService, create_beam_search_service

class FallbackASRService:
    """
    Fallback ASR Service using Wav2Vec 2.0 ONNX.
    Used for noisy audio or low-confidence segments.
    """
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or settings.fallback_asr_model_path)
        self.onnx_path = self.model_dir / "model_int8.onnx"
        
        if not self.onnx_path.exists():
            print(f"⚠️ Fallback ASR model not found at {self.onnx_path}")
            self.session = None
            return

        print(f"🚀 Initializing FallbackASRService with model at {self.model_dir}")
        
        # 1. Setup Session Options to reduce log noise
        opts = ort.SessionOptions()
        opts.log_severity_level = 3 # Warning and above only
        
        # 2. Provider Selection
        # We use CPUExecutionProvider by default for the fallback service 
        # to avoid the "Context leak detected" log spam emitted by CoreML on macOS.
        providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(
                str(self.onnx_path), 
                sess_options=opts,
                providers=providers
            )
            print(f"✅ Fallback ASR Session loaded with providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"⚠️ Fallback ASR initialization failed: {e}")
            self.session = None
        
        # Load processor from local files
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(str(self.model_dir), local_files_only=True)
        except Exception as e:
            print(f"⚠️ Failed to load processor locally: {e}. Attempting standard load...")
            self.processor = Wav2Vec2Processor.from_pretrained(str(self.model_dir))
            
        # Initialize Beam Search with Bilingual KenLM
        kenlm_vn = settings.kenlm_vn_path
        kenlm_en = settings.kenlm_en_path
        vocab_path = str(self.model_dir / "vocab.json")
        self.beam_search = create_beam_search_service(vocab_path, kenlm_vn, kenlm_en)
        
        # Get labels from tokenizer
        self.labels = [self.processor.tokenizer.decode([i]) for i in range(len(self.processor.tokenizer.get_vocab()))]

    def transcribe(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Transcribes audio using Wav2Vec + Beam Search decoding.
        """
        if self.session is None:
            return {"text": "", "confidence": 0.0, "error": "Model not loaded"}

        start_time = time.time()
        
        # Preprocess
        inputs = self.processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="np",
            padding=True
        )
        input_values = inputs["input_values"].astype(np.float32)
        
        # Inference
        outputs = self.session.run(None, {"input_values": input_values})
        logits = outputs[0][0] # (time, vocab_size)
        
        # Beam Search Decoding
        results = self.beam_search.decode(logits, self.labels)
        
        if not results:
            return {"text": "", "confidence": 0.0, "latency_ms": 0}
            
        # results[0] is now a Dict with {"text", "score", "lang"}
        best_result = results[0]
        total_score = best_result["score"]
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "text": best_result["text"],
            "confidence": float(np.exp(total_score) if total_score < 0 else 1.0),
            "latency_ms": latency_ms,
            "engine": "Wav2Vec-Fallback",
            "language": best_result["lang"]
        }

# Singleton instance
fallback_asr_service = FallbackASRService()
