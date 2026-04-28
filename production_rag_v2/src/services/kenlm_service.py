import os
import kenlm
from typing import List, Tuple, Optional, Dict, Any
from src.core.config import settings

class KenLMService:
    """
    Service for language model rescoring using KenLM.
    Evaluates the probability of generated text sequences in VN and EN.
    """
    def __init__(self, vn_model_path: Optional[str] = None, en_model_path: Optional[str] = None):
        self.vn_model_path = vn_model_path or settings.kenlm_vn_path
        self.en_model_path = en_model_path or settings.kenlm_en_path
        
        self.vn_model = self._load_model(self.vn_model_path, "Vietnamese")
        self.en_model = self._load_model(self.en_model_path, "English")

    def _load_model(self, path: str, label: str):
        if path and os.path.exists(path):
            try:
                model = kenlm.Model(path)
                print(f"✅ KenLM {label} model loaded from {path}")
                return model
            except Exception as e:
                print(f"❌ Failed to load KenLM {label} model: {e}")
        else:
            print(f"⚠️ KenLM {label} model file missing at {path}")
        return None

    def get_score(self, sentence: str, lang: str = "vn") -> float:
        """Calculates the log-probability for a specific language."""
        model = self.vn_model if lang == "vn" else self.en_model
        if model is None:
            return -100.0 # High penalty for missing model
        return model.score(sentence, bos=True, eos=True)

    def get_bilingual_score(self, sentence: str) -> Dict[str, Any]:
        """
        Scores the sentence against both models and returns the best fit.
        Applies length normalization.
        """
        if not sentence.strip():
            return {"score": -100.0, "lang": "unknown", "confidence": 0.0}

        # Basic length normalization (average log-prob per word)
        # Add 1 to avoid division by zero and provide slight smoothing
        words = sentence.strip().split()
        num_words = len(words)
        
        vn_raw = self.get_score(sentence, lang="vn")
        en_raw = self.get_score(sentence, lang="en")
        
        vn_norm = vn_raw / (num_words + 0.1)
        en_norm = en_raw / (num_words + 0.1)
        
        if vn_norm >= en_norm:
            return {"score": vn_raw, "lang": "vi", "confidence": vn_norm}
        else:
            return {"score": en_raw, "lang": "en", "confidence": en_norm}

    def rescore_hypotheses(self, hypotheses: List[str]) -> List[Dict[str, Any]]:
        """
        Scores a list of hypotheses using bilingual logic and returns them sorted.
        """
        if not hypotheses:
            return []
            
        scored = []
        for h in hypotheses:
            result = self.get_bilingual_score(h)
            result["text"] = h
            scored.append(result)
            
        # Sort by score descending
        return sorted(scored, key=lambda x: x["score"], reverse=True)

# Singleton
kenlm_service = KenLMService()
