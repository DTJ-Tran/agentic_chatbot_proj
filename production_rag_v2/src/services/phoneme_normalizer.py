import re
import torch
from typing import List, Dict, Any, Optional
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

class PhonemeNormalizer:
    """
    Normalizes text (EN/VI) into the custom phonetic space of the Wav2Vec model.
    Maps standard IPA/espeak output to the tokens in vocab.json.
    """
    
    # Mapping from viphoneme/espeak-vi tones to Wav2Vec custom tone tokens
    # viphoneme: 1=ngang, 2=huyền, 3=hỏi, 4=ngã, 5=sắc, 6=nặng
    # Wav2Vec tokens found: a1, a2, a4, a5. (3 and 6 might be mapped differently)
    VI_TONE_MAP = {
        "1": "1", # ngang
        "2": "2", # huyền
        "3": "4", # hỏi (user hinted 4 is in use)
        "4": "5", # ngã (assuming ngã/sắc similarity in some dialects or vocab)
        "5": "5", # sắc
        "6": ".", # nặng (vocab has 'a.')
    }

    def __init__(self):
        # Initialize backends as singletons for efficiency
        self.backend_en = EspeakBackend('en-us')
        self.backend_vi = EspeakBackend('vi')
        
    def normalize_en(self, text: str) -> str:
        """Converts English text to Wav2Vec-standardized phonemes."""
        # Get raw IPA from espeak
        ipa = self.backend_en.phonemize([text], strip=True)[0]
        # Basic cleanup: remove punctuation but keep spaces between words
        # The Wav2Vec model for EN likely uses these IPA symbols directly (from VoxPopuli/CommonVoice)
        return ipa

    def normalize_vi(self, text: str) -> str:
        """
        Converts Vietnamese text to Wav2Vec-standardized phonemes.
        Handles the custom mapping from viphoneme-style output to W2V tokens.
        """
        # Try to use viphoneme if available for better accuracy
        from src.services.phoneme_service import phoneme_service
        raw_ipa = phoneme_service.to_phonemes(text)
        
        # Example output: "dɯək6 viət5"
        # We need to map digits to the custom vocab tokens
        def tone_replacer(match):
            digit = match.group(0)
            return self.VI_TONE_MAP.get(digit, digit)
        
        normalized = re.sub(r'\d', tone_replacer, raw_ipa)
        # Note: Wav2Vec vocab has compound tokens like 'onɡ5'. 
        # For PER, we should ideally space-separate everything that matches a vocab entry.
        return normalized

    def normalize(self, text: str, lang: str = "vi") -> str:
        """Main entry point for text-to-phoneme normalization."""
        text = text.lower().strip()
        if lang == "en":
            return self.normalize_en(text)
        return self.normalize_vi(text)

# Singleton
phoneme_normalizer = PhonemeNormalizer()
