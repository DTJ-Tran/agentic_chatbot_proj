import os
import torch
import torchaudio
import torchaudio.functional as F
import onnxruntime as ort
from transformers import AutoTokenizer, WhisperFeatureExtractor
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from unidecode import unidecode
from src.core.config import settings

logger = logging.getLogger(__name__)

class QwenAlignService:
    """
    High-precision word-level aligner using the local Qwen-3-ForcedAligner ONNX model.
    Optimized for Apple Silicon via ONNX Runtime.
    """
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or settings.qwen_align_model_path
        self.model_path = os.path.join(self.model_dir, "model.onnx")
        
        # 1. Initialize Components Manually for Reliability
        logger.info(f"🚀 Initializing Qwen Components from {model_dir}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 2. Build Alignment Vocabulary Map (IDs < 5000)
        # Most single characters are in the first few hundred IDs.
        self.align_vocab = {}
        vocab = self.tokenizer.get_vocab()
        for token, idx in vocab.items():
            if idx < 5000:
                self.align_vocab[token] = idx
        
        # 3. Initialize ONNX Session
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"✅ Qwen ONNX Session loaded with providers: {self.session.get_providers()}")
        except Exception as e:
            logger.warning(f"⚠️ CoreML failed, falling back to standard CPU: {e}")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

    def _get_alignment_targets(self, text: str) -> List[int]:
        """
        Maps the text string to a sequence of character IDs within the [0, 5000) range.
        Uses unidecode as fallback for characters outside the aligner's vocabulary.
        """
        targets = []
        # Qwen's space is often ID 220 (Ġ) or 151644, but we need the one < 5000
        space_id = self.align_vocab.get("Ġ", 220) 
        
        for char in text:
            if char == " ":
                targets.append(space_id)
                continue
                
            # Try exact match (Latin-1, basic characters)
            idx = self.align_vocab.get(char)
            if idx is not None:
                targets.append(idx)
            else:
                # Fallback: Strip accents (e.g. 'ố' -> 'o')
                # Most base Latin characters are < 100
                simple_char = unidecode(char)
                for sc in simple_char:
                    idx = self.align_vocab.get(sc, 0) # Use 0 (blank/unk) as ultimate fallback
                    targets.append(idx)
        return targets

    async def align(self, audio: np.ndarray, text: str, language: str = "en", sr: int = 16000) -> List[Dict[str, Any]]:
        """
        Aligns the transcript to the audio using the Qwen-3 Forced Aligner.
        """
        if not text.strip():
            return []

        # 1. Pre-process Audio & Text
        # We need len(input_ids) == len(targets) to satisfy CTC constraints
        # and all IDs MUST be < 5000 for the head.
        target_ids = self._get_alignment_targets(text)
        input_ids = np.array([target_ids], dtype=np.int64)
        
        # Audio Features
        audio_inputs = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = audio_inputs.input_features.numpy().astype(np.float32)
        
        # Binary masks
        attention_mask = np.ones((1, input_ids.shape[1]), dtype=np.int64)
        feature_attention_mask = np.ones((1, input_features.shape[2]), dtype=np.int32)

        # 2. Run ONNX Inference
        ort_inputs = {
            "input_ids": input_ids,
            "input_features": input_features,
            "attention_mask": attention_mask,
            "feature_attention_mask": feature_attention_mask
        }
        
        try:
            outputs = self.session.run(None, ort_inputs)
            logits = outputs[0] # [batch, tokens, 5000]
        except Exception as e:
            logger.error(f"❌ Qwen Inference failed: {e}")
            return []

        # 3. Perform Forced Alignment (Trellis search)
        # Logits sequence length matches input_ids length (1 logit per token)
        emission_log_probs = torch.log_softmax(torch.from_numpy(logits), dim=-1)
        
        # Use the SAME IDs as targets
        targets = torch.tensor([target_ids], dtype=torch.int32)
        
        # Blank / Pad ID 0 is safest
        pad_token_id = 0 
        
        try:
            # path maps each frame (time) to a target index (1-based)
            # wait, the frames are NOT the time axis here?
            # In Qwen-3 ForcedAligner ONNX, the sequence length of logits is AFTER the attention.
            # Usually, the model produces one frame of time per feature sequence, 
            # NOT one frame per token. 
            
            # RE-CHECK: If logits.shape[1] matches input_ids.shape[1], it's per-token.
            # If so, ctc alignment finds which input token matches which audio segment.
            path, scores = F.forced_align(emission_log_probs, targets, blank=pad_token_id)
            path = path[0] 
        except Exception as e:
            logger.error(f"❌ CTC Alignment failed: {e}. Logits shape: {logits.shape}, Targets len: {len(target_ids)}")
            return []

        # 4. Convert Path to Timestamps
        num_logit_frames = logits.shape[1]
        audio_duration = len(audio) / sr
        time_per_frame = audio_duration / num_logit_frames
        
        # 5. Map Character Aligments to Words
        # 'path' gives the target index for each frame.
        char_timestamps = []
        current_target_idx = -1
        last_frame = 0
        
        target_chars = []
        # Reconstruct exactly what we aligned
        for char in text:
            if char == " ":
                target_chars.append(" ")
            else:
                simple_char = unidecode(char)
                for sc in simple_char:
                    target_chars.append(char) # Keep original char for merging

        for i, t_idx in enumerate(path):
            if t_idx != 0: # Not blank
                if t_idx != int(current_target_idx):
                    if current_target_idx != -1:
                        char_timestamps.append({
                            "char": target_chars[int(current_target_idx)-1],
                            "start": last_frame * time_per_frame,
                            "end": i * time_per_frame
                        })
                    current_target_idx = t_idx
                    last_frame = i

        if current_target_idx != -1:
            char_timestamps.append({
                "char": target_chars[int(current_target_idx)-1],
                "start": last_frame * time_per_frame,
                "end": num_logit_frames * time_per_frame
            })

        # Group character timestamps into words
        word_segments = []
        current_word = ""
        word_start = None
        
        for ct in char_timestamps:
            if ct["char"] == " ":
                if current_word:
                    word_segments.append({
                        "word": current_word,
                        "start": word_start,
                        "end": last_end
                    })
                    current_word = ""
                    word_start = None
                continue
            
            if word_start is None:
                word_start = ct["start"]
            current_word += ct["char"]
            last_end = ct["end"]

        if current_word:
            word_segments.append({
                "word": current_word,
                "start": word_start,
                "end": last_end
            })

        return word_segments

# Singleton
qwen_aligner = QwenAlignService()
