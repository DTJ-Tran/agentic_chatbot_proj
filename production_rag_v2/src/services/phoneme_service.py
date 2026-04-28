import re
from typing import List, Dict, Any

try:
    import viphoneme
    from viphoneme import vi2IPA, vi2IPA_split
    import vinorm
    from num2words import num2words
    VIFONEME_AVAILABLE = True
except ImportError:
    VIFONEME_AVAILABLE = False

class PhonemeService:
    """
    Service for Vietnamese Phoneme mapping and normalization.
    Uses the Viphoneme library for G2P transformation with a Mac-compatible monkeypatch.
    """
    def __init__(self):
        self.vowel_map = {
            'a': 'a', 'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'aw', 'ằ': 'aw', 'ắ': 'aw', 'ẳ': 'aw', 'ẵ': 'aw', 'ặ': 'aw',
            'â': 'aa', 'ầ': 'aa', 'ấ': 'aa', 'ẩ': 'aa', 'ẫ': 'aa', 'ậ': 'aa',
            'e': 'e', 'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'ee', 'ề': 'ee', 'ế': 'ee', 'ể': 'ee', 'ễ': 'ee', 'ệ': 'ee',
            'i': 'i', 'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'o': 'o', 'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'oo', 'ồ': 'oo', 'ố': 'oo', 'ổ': 'oo', 'ỗ': 'oo', 'ộ': 'oo',
            'ơ': 'ow', 'ờ': 'ow', 'ớ': 'ow', 'ở': 'ow', 'ỡ': 'ow', 'ợ': 'ow',
            'u': 'u', 'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'uw', 'ừ': 'uw', 'ứ': 'uw', 'ử': 'uw', 'ữ': 'uw', 'ự': 'uw',
            'y': 'i', 'ỳ': 'i', 'ý': 'i', 'ỷ': 'i', 'ỹ': 'i', 'ỵ': 'i',
        }
        if VIFONEME_AVAILABLE:
            self._monkeypatch_vinorm()

    def _monkeypatch_vinorm(self):
        """
        Monkeypatch vinorm.TTSnorm to avoid ELF binary issues on Mac.
        Provides a Python-based normalization for numbers and dates.
        """
        def safe_norm(text, **kwargs):
            # Basic regex-based normalization for numbers and dates
            import re
            
            # 1. Dates: dd/mm/yyyy or dd/mm
            def replace_date(match):
                d, m = match.group(1), match.group(2)
                y = match.group(3) if match.group(3) else None
                try:
                    res = f"ngày {num2words(int(d), lang='vi')} tháng {num2words(int(m), lang='vi')}"
                    if y:
                        res += f" năm {num2words(int(y), lang='vi')}"
                    return res
                except:
                    return match.group(0)

            text = re.sub(r'(\d{1,2})/(\d{1,2})(?:/(\d{4}))?', replace_date, text)

            # 2. Numbers
            def replace_num(match):
                try:
                    return num2words(int(match.group(0)), lang='vi')
                except:
                    return match.group(0)

            text = re.sub(r'\d+', replace_num, text)
            
            # Simple cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Replace the broken TTSnorm in both namespaces
        vinorm.TTSnorm = safe_norm
        if hasattr(viphoneme, 'TTSnorm'):
            viphoneme.TTSnorm = safe_norm
        print("DEBUG: PhonemeService - vinorm.TTSnorm monkeypatched for Mac compatibility.")

    def to_phonemes(self, text: str) -> str:
        """Converts Vietnamese text to IPA phonemic representation."""
        if VIFONEME_AVAILABLE:
            try:
                # vi2IPA returns a string like "dɯək6 viət5"
                return vi2IPA(text)
            except Exception as e:
                print(f"DEBUG: Viphoneme failed: {e}")
        
        # Fallback to simplistic logic
        return self._fallback_to_phonemes(text)

    def to_phoneme_sequence(self, text: str, delimit: str = "|") -> list[str]:
        """Converts text to an IPA sequence split by a delimiter."""
        if VIFONEME_AVAILABLE:
            try:
                # Use a regex-safe delimiter internally because viphoneme uses it in re.sub(delimit+'+', ...)
                safe_delimit = " " # Space is safe and already used in some parts of the library
                ipa_str = vi2IPA_split(text, safe_delimit)
                # Cleanup and convert to list using the user's requested delimiter if needed, 
                # but here we just return a list.
                return [p.strip() for p in ipa_str.split(safe_delimit) if p.strip()]
            except Exception as e:
                print(f"DEBUG: Viphoneme split failed: {e}")
        
        # Fallback to character-level mapping
        fallback = self._fallback_to_phonemes(text)
        return list(fallback.replace(" ", ""))

    def _fallback_to_phonemes(self, text: str) -> str:
        text = text.lower()
        phonemes = []
        for char in text:
            if char in self.vowel_map:
                phonemes.append(self.vowel_map[char])
            elif re.match(r'[a-z]', char):
                phonemes.append(char)
            elif char.isspace():
                phonemes.append(' ')
        return "".join(phonemes)

    def normalize_phonemes(self, phoneme_str: str) -> str:
        """Normalizes phonemic strings for Trie matching."""
        # Remove consecutive duplicates (standard CTC practice)
        return re.sub(r'(.)\1+', r'\1', phoneme_str)

# Singleton
phoneme_service = PhonemeService()
