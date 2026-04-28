import re
import unicodedata
from typing import List, Optional

def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFKC and handle basic whitespace cleanup."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x0c", " ")
    return text

def clean_whitespace(text: str) -> str:
    """Collapse multiple spaces and normalize newlines."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\d+\s*\n\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(\.\s*){2,}", ". ", text)
    text = re.sub(r"(-{2,})", " ", text)
    return text.strip()

def strip_markdown(text: str) -> str:
    """Remove common markdown artifacts."""
    if not text:
        return ""
    text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)
    text = re.sub(r'<https?://[^>\s]+>', '', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\|', ' ', text)
    return text

def clean_noise(text: str) -> str:
    """Filter out common marketing noise."""
    if not text:
        return ""
    filtered_lines = []
    noise_patterns = ["connect with us", "copyright", "©", "all rights reserved", "subscribe"]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped: continue
        lower_line = stripped.lower()
        if any(pattern in lower_line for pattern in noise_patterns): continue
        if len(stripped) < 40 and stripped.isupper() and len(stripped.split()) <= 4: continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()

def detect_language(text: str) -> str:
    """Heuristic logic to detect 'vi' vs 'en'."""
    if not text or len(text) < 10: return "en"
    text_lower = text.lower()
    vi_diacritics = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ"
    vi_char_count = sum(1 for c in text_lower if c in vi_diacritics)
    vi_stopwords = {"và", "là", "của", "trong", "một", "những", "các"}
    vi_word_hits = sum(1 for word in text_lower.split() if word in vi_stopwords)
    en_stopwords = {"the", "is", "are", "of", "in", "and"}
    en_word_hits = sum(1 for word in text_lower.split() if word in en_stopwords)
    if vi_char_count > 3 or vi_word_hits > en_word_hits: return "vi"
    return "en"

def full_clean(text: str) -> str:
    """Standard production pipeline for text cleansing."""
    text = normalize_unicode(text)
    text = strip_markdown(text)
    text = clean_whitespace(text)
    text = clean_noise(text)
    return text
