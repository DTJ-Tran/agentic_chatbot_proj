import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.services.trie_service import LexiconTrie
from src.services.kenlm_service import KenLMService

class BeamSearchService:
    """
    Service for Beam Search decoding of CTC logits.
    Integrates Lexicon Trie and KenLM rescoring.
    """
    def __init__(self, trie: LexiconTrie, kenlm_service: KenLMService, beam_size: int = 5):
        self.trie = trie
        self.kenlm_service = kenlm_service
        self.beam_size = beam_size

    def decode(self, logits: np.ndarray, labels: List[str]) -> List[Tuple[str, float]]:
        """
        Decodes CTC logits using beam search.
        logits: (time, num_labels)
        labels: list of characters/phonemes corresponding to logit indices
        """
        # Simplified Beam Search implementation
        # In a production setting, this would be a more optimized C++/Cython implementation
        
        # Initial beam: empty string with score 0
        beams = [("", 0.0)]
        
        for t in range(logits.shape[0]):
            new_beams = []
            # Get top probabilities for current timestep
            probs = np.exp(logits[t])
            top_indices = np.argsort(probs)[-self.beam_size:]
            
            for prefix, score in beams:
                for idx in top_indices:
                    char = labels[idx]
                    if char == '[PAD]': # CTC Blank
                        new_beams.append((prefix, score + np.log(probs[idx])))
                    else:
                        new_prefix = prefix + char
                        # Optional: Trie pruning
                        # if self.trie.search(new_prefix):
                        new_beams.append((new_prefix, score + np.log(probs[idx])))
            
            # Sort and prune new beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_size]
            
        # Final rescoring with KenLM
        final_results = []
        for text, am_score in beams:
            # Clean up CTC duplicates and blanks
            decoded_text = self._ctc_postprocess(text)
            lm_result = self.kenlm_service.get_bilingual_score(decoded_text)
            
            # Weighted total: AM + LM
            # Weight is scaled by language (EN often needs slightly lower weight in CTC)
            lm_weight = 0.10 if lm_result["lang"] == "en" else 0.15
            total_score = am_score + lm_weight * lm_result["score"]
            
            final_results.append({
                "text": decoded_text, 
                "score": total_score,
                "lang": lm_result["lang"]
            })
            
        return sorted(final_results, key=lambda x: x["score"], reverse=True)

    def _ctc_postprocess(self, text: str) -> str:
        """Removes duplicate characters and handles blanks."""
        if not text:
            return ""
        # Remove consecutive duplicates
        res = [text[0]]
        for i in range(1, len(text)):
            if text[i] != text[i-1]:
                res.append(text[i])
        return "".join(res).strip()

# Factory function to create BeamSearchService from config
def create_beam_search_service(vocab_path: str, kenlm_vn_path: Optional[str] = None, kenlm_en_path: Optional[str] = None) -> BeamSearchService:
    from src.services.trie_service import load_lexicon_from_file
    from src.services.kenlm_service import KenLMService
    
    trie = load_lexicon_from_file(vocab_path)
    kenlm = KenLMService(kenlm_vn_path, kenlm_en_path)
    return BeamSearchService(trie, kenlm)
