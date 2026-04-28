import jiwer
from typing import Dict, Any, List

class PEREvaluator:
    """
    Calculates Phoneme Error Rate (PER) and generates LLM routing flags.
    Decomposes errors into Substitutions, Insertions, and Deletions.
    """
    
    def __init__(self, per_threshold: float = 0.30, tiebreaker_threshold: float = 0.15):
        self.per_threshold = per_threshold
        self.tiebreaker_threshold = tiebreaker_threshold

    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Calculates PER and error counts. 
        Strings should be space-separated phoneme sequences.
        """
        # Ensure input is not empty to avoid division by zero errors in jiwer
        if not reference.strip():
            return {
                "per": 1.0 if hypothesis.strip() else 0.0,
                "hits": 0,
                "substitutions": 0,
                "insertions": len(hypothesis.split()) if hypothesis.strip() else 0,
                "deletions": 0
            }
            
        result = jiwer.process(reference, hypothesis)
        
        # jiwer's 'wer' is used as PER here
        per = result.wer
        
        # Extract operations
        # Note: hits is calculated as reference_length - substitutions - deletions
        ref_len = len(reference.split())
        sub = result.substitutions
        ins = result.insertions
        dele = result.deletions
        hits = ref_len - sub - dele
        
        return {
            "per": round(float(per), 4),
            "error_breakdown": {
                "hits": int(hits),
                "substitutions": int(sub),
                "insertions": int(ins),
                "deletions": int(dele)
            }
        }

    def generate_routing_flags(self, moon_metrics: Dict[str, Any], wav_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """
        Applies deterministic logic for LLM flagging using Ground Truth PER.
        """
        moon_per = moon_metrics["per"]
        wav_per = wav_metrics["per"]
        moon_err = moon_metrics["error_breakdown"]
        wav_err = wav_metrics["error_breakdown"]
        
        requires_rewrite = (moon_per > self.per_threshold) or (wav_per > self.per_threshold)
        req_vocab_check = False
        for err in [moon_err, wav_err]:
            if err["deletions"] == 0 and err["insertions"] == 0 and err["substitutions"] > 2:
                req_vocab_check = True
                break
        req_tiebreaker = abs(moon_per - wav_per) > self.tiebreaker_threshold
        
        return {
            "requires_llm_rewrite": requires_rewrite,
            "requires_llm_vocab_check": req_vocab_check,
            "requires_llm_tiebreaker": req_tiebreaker,
            "heuristic_based": False
        }

    # Common ASR Hallucination templates (closing phrases)
    HALLUCINATION_TEMPLATES = {
        "cảm ơn các bạn", "theo dõi", "hẹn gặp lại", 
        "thank you for watching", "thanks for watching",
        "see you next time", "cảm ơn anh", "hallucination", 
        "cảm ơn đã theo dõi", "hẹn gặp lại các bạn"
    }

    def generate_heuristic_flags(self, moon_phonemes: str, wav_phonemes: str, 
                                 moon_orig: str, moon_stretched: str,
                                 duration: float = 0.0) -> Dict[str, Any]:
        """
        Flags segments for LLM review based on model divergence (No Ground Truth needed).
        Treats Wav2Vec as the Pseudo-Ground Truth for Moonshine.
        Includes "Hallucination Outlier" detection for short segments or common generic templates.
        """
        # 1. Cross-Model Divergence (Wav2Vec as Reference)
        cross_per = 1.0
        if moon_phonemes and wav_phonemes:
            # We use jiwer.wer but logically it represents PER here
            cross_per = jiwer.wer(wav_phonemes, moon_phonemes)
        
        # 2. Intra-Model Stability (Moonshine Original vs. Stretched)
        stability_divergence = 0.0
        if moon_orig and moon_stretched:
            stability_divergence = jiwer.wer(moon_orig, moon_stretched)
        
        # 3. Hallucination Outlier Detection
        moon_tokens = moon_phonemes.split() if moon_phonemes else []
        wav_tokens = wav_phonemes.split() if wav_phonemes else []
        tdr = len(moon_tokens) / max(0.1, duration)
        
        # A: Density-based (Very short audio + too many tokens)
        is_density_hallu = (duration > 0 and duration < 1.2 and tdr > 8.0)
        
        # B: Semantic-based (Common closing phrases when Wav2Vec has more info)
        is_semantic_hallu = False
        moon_lower = (moon_orig or "").lower()
        has_template = any(t in moon_lower for t in self.HALLUCINATION_TEMPLATES)
        
        # If it matches a template AND Wav2Vec found significantly more unique sounds (entropy)
        if has_template and len(wav_tokens) > len(moon_tokens) * 1.2:
            is_semantic_hallu = True
            
        is_hallucination = is_density_hallu or is_semantic_hallu
        
        # Heuristic Thresholds
        requires_rewrite = cross_per > 0.40 or stability_divergence > 0.50
        
        # OUTLIER SUPPRESSION: If it's a hallucination, do NOT send to LLM
        if is_hallucination:
            requires_rewrite = False
            
        requires_tiebreaker = (stability_divergence > 0.15 or abs(cross_per) > 0.50) and not is_hallucination
        
        return {
            "requires_llm_rewrite": requires_rewrite,
            "requires_llm_vocab_check": False,
            "requires_llm_tiebreaker": requires_tiebreaker,
            "is_hallucination": is_hallucination,
            "hallucination_type": "density" if is_density_hallu else "semantic" if is_semantic_hallu else None,
            "heuristic_based": True,
            "cross_per": round(cross_per, 3),
            "debug_metrics": {
                "stability_div": round(stability_divergence, 3),
                "tdr": round(tdr, 2),
                "token_ratio": round(len(wav_tokens)/max(1, len(moon_tokens)), 2)
            }
        }

# Singleton
per_evaluator = PEREvaluator()
