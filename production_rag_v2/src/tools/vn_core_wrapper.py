import py_vncorenlp
import os
from pathlib import Path
from typing import Optional, List

class VnCoreNLPWrapper:
    """
    Thread-safe, lazy-loading singleton wrapper for the VnCoreNLP word segmenter.
    """
    _instance: Optional['VnCoreNLPWrapper'] = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VnCoreNLPWrapper, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_dir: Optional[str] = None):
        if not hasattr(self, 'initialized'):
            # Default directory in the root of the project
            if model_dir is None:
                # __file__ is at .../production_rag_v2/src/tools/vn_core_wrapper.py
                v2_root = Path(__file__).parent.parent.parent
                self.model_dir = str(v2_root / "resources" / "VnCoreNLP")
            else:
                self.model_dir = model_dir
                
            self.initialized = True

    def warm_up(self):
        """Eagerly triggers the loading of the VnCoreNLP model."""
        if self._model is None:
            print("📦 [VnCoreNLP] Warming up word segmenter (Java)...")
            try:
                self._get_model()
                print("✅ [VnCoreNLP] Model ready.")
            except Exception as e:
                print(f"⚠️ [VnCoreNLP] Warm up failed: {e}")

    def _get_model(self):
        """Lazy loader for the VnCoreNLP model."""
        if self._model is None:
            # Check v2 root first
            os.makedirs(self.model_dir, exist_ok=True)
            
            # If v2 model is missing, check v1 sibling for existing files
            if not os.listdir(self.model_dir):
                v1_model_dir = str(Path(self.model_dir).parent.parent.parent / "resources" / "VnCoreNLP")
                if os.path.exists(v1_model_dir) and os.listdir(v1_model_dir):
                    print(f"🔗 [VnCoreNLP] Model found in v1 sibling. Linking...")
                    self.model_dir = v1_model_dir
                else:
                    print(f"📥 [VnCoreNLP] Model missing globally. Downloading to {self.model_dir}...")
                    py_vncorenlp.download_model(save_dir=self.model_dir)
            
            self._model = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "ner"], 
                save_dir=self.model_dir
            )
        return self._model

    def segment(self, text: str) -> str:
        """Segment Vietnamese text."""
        if not text or not text.strip():
            return ""
        try:
            model = self._get_model()
            sentences = model.word_segment(text)
            return " ".join(sentences)
        except Exception as e:
            print(f"VnCoreNLP Segmentation Error: {e}")
            return text

    def segment_sentences(self, text: str) -> List[str]:
        """Returns a list of segmented sentences."""
        if not text or not text.strip():
            return []
        try:
            model = self._get_model()
            return model.word_segment(text)
        except Exception as e:
            print(f"VnCoreNLP Segmentation Error: {e}")
            return [text]
