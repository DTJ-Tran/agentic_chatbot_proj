import os
# Set local cache directory for Hugging Face to avoid permission issues
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "tmp", "huggingface")

import torch
import torchaudio
import torchaudio.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class NativeAlignService:
    """
    A high-performance word-level aligner that runs natively in Python using Wav2Vec2.
    Eliminates Docker overhead for real-time applications.
    """
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        logger.info(f"🚀 NativeAlignService initialized on {self.device}")

    def _get_model(self, language: str):
        """Lazy-loads the appropriate Wav2Vec2 model for the language."""
        if language in self.models:
            return self.models[language], self.processors[language]

        if language == "vi":
            model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
        else:
            model_id = "facebook/wav2vec2-base-960h"

        logger.info(f"📥 Loading alignment model for {language}: {model_id}")
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device).eval()
        
        self.models[language] = model
        self.processors[language] = processor
        return model, processor

    async def align(self, audio: np.ndarray, text: str, language: str = "en", sr: int = 16000) -> List[Dict[str, Any]]:
        """
        Aligns the transcript to the audio using forced alignment.
        """
        if not text.strip():
            return []

        model, processor = self._get_model(language)
        
        # 1. Pre-process audio
        waveform = torch.from_numpy(audio).float().to(self.device)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # 2. Tokenize text (lowercase for CTC models)
        # We need to treat the text as tokens that match the model's vocabulary
        clean_text = text.upper() if language == "en" else text.lower()
        
        # Tokenize and get token IDs
        tokens = processor.tokenizer(clean_text, return_tensors="pt").input_ids[0]
        # Remove special tokens added by tokenizer if any (like <s>, </s>)
        tokens = [t for t in tokens.tolist() if t not in [processor.tokenizer.cls_token_id, processor.tokenizer.sep_token_id, processor.tokenizer.pad_token_id]]
        
        if not tokens:
            return []

        # 3. Get Emissions (Probabilities over time)
        with torch.no_grad():
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
            emissions = model(inputs.input_values).logits
            # emissions: (batch, time, tokens)
        
        # 4. Perform Alignment
        # Convert emissions to log probabilities
        emission_log_probs = torch.log_softmax(emissions, dim=-1)
        
        # Prepare targets (tokens) for torchaudio.forced_align
        targets = torch.tensor([tokens], device=self.device)
        
        try:
            # alignment_path: (N,) indices of tokens at each time step
            # score: confidence
            # Note: torchaudio.functional.forced_align performs the trellis DP
            # Each step in the path is a frame.
            path, scores = F.forced_align(emission_log_probs, targets, blank=processor.tokenizer.pad_token_id or 0)
            path = path[0] # remove batch dim
        except Exception as e:
            logger.error(f"❌ Alignment failed: {e}")
            return []

        # 5. Convert Path to Timestamps
        # The number of frames matches the emission length
        num_frames = emission_log_probs.shape[1]
        time_per_frame = len(audio) / sr / num_frames
        
        # Segment the path into words
        # This is a simplified merge: we know where the tokens start and end in the path
        # CTC path elements are: [token_index, token_index, ...]
        # We need to identify transitions
        
        word_segments = []
        current_token_idx = -1
        start_frame = 0
        
        # Map token IDs back to strings for reconstruction
        all_tokens = [processor.tokenizer.decode([t]) for t in tokens]
        
        # Build word boundaries by looking for transitions in the path
        # 0 in path usually means blank token in CTC
        
        token_frames = []
        for i, t_idx in enumerate(path):
            if t_idx != 0: # Not a blank
                if t_idx != current_token_idx:
                    # New token started
                    if current_token_idx != -1:
                        token_frames.append({
                            "token": processor.tokenizer.decode([tokens[current_token_idx-1]]),
                            "start": start_frame * time_per_frame,
                            "end": i * time_per_frame
                        })
                    current_token_idx = t_idx
                    start_frame = i
        
        # Add last token
        if current_token_idx != -1:
            token_frames.append({
                "token": processor.tokenizer.decode([tokens[current_token_idx-1]]),
                "start": start_frame * time_per_frame,
                "end": len(path) * time_per_frame
            })

        # Group tokens into words
        # Most Wav2Vec2 models use '|' or ' ' as word separators
        word_sep = "|" if language == "en" else " "
        
        current_word = []
        word_start = None
        
        for tf in token_frames:
            if tf["token"] == word_sep:
                if current_word:
                    word_segments.append({
                        "word": "".join(current_word),
                        "start": word_start,
                        "end": last_end
                    })
                    current_word = []
                    word_start = None
            else:
                if word_start is None:
                    word_start = tf["start"]
                current_word.append(tf["token"])
                last_end = tf["end"]
        
        # Final word
        if current_word:
            word_segments.append({
                "word": "".join(current_word),
                "start": word_start,
                "end": last_end
            })

        return word_segments

# Singleton
native_aligner = NativeAlignService()
