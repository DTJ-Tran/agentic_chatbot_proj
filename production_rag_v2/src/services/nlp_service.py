import py_vncorenlp
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
from src.core.config import settings

class NLPService:
    """
    Provides NLP enrichment for meeting transcripts.
    Uses VNCoreNLP for Vietnamese processing.
    """
    def __init__(self, vncorenlp_dir: Optional[str] = None):
        self.vncorenlp_dir = vncorenlp_dir or settings.vncorenlp_path
        
    def warm_up(self):
        """Pre-loads the VnCoreNLP model."""
        from src.tools.vn_core_wrapper import VnCoreNLPWrapper
        VnCoreNLPWrapper(self.vncorenlp_dir).warm_up()

    @property
    def model(self):
        from src.tools.vn_core_wrapper import VnCoreNLPWrapper
        return VnCoreNLPWrapper(self.vncorenlp_dir)._get_model()

    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        Extracts named entities from the text.
        Focuses on Persons (PER), Organizations (ORG), and Locations (LOC).
        """
        if not text.strip():
            return {"PER": set(), "ORG": set(), "LOC": set()}

        annotations = self.model.annotate_text(text)
        
        entities = {
            "PER": set(),
            "ORG": set(),
            "LOC": set()
        }

        # VnCoreNLP annotations structure: { '0': [ { 'wordForm':..., 'nerLabel':... }, ... ], ... }
        for sent_idx in annotations:
            for word in annotations[sent_idx]:
                label = word.get("nerLabel", "O")
                form = word.get("wordForm", "").replace("_", " ")
                
                if label.endswith("-PER"):
                    entities["PER"].add(form)
                elif label.endswith("-ORG"):
                    entities["ORG"].add(form)
                elif label.endswith("-LOC"):
                    entities["LOC"].add(form)
                    
        return entities

    def normalize_transcript(self, text: str) -> str:
        """
        Performs basic text normalization and word segmentation for Vietnamese.
        """
        segmented_sentences = self.model.word_segment(text)
        return " ".join(segmented_sentences).replace("_", " ")

    def extract_summary_context(self, text: str) -> Dict[str, Any]:
        """
        Bundles transcripts with extracted metadata.
        """
        entities = self.extract_entities(text)
        # Convert sets to lists for JSON serialization
        return {
            "participants": list(entities["PER"]),
            "organizations": list(entities["ORG"]),
            "locations": list(entities["LOC"]),
            "segment_count": len(text.split()) // 20 # heuristic
        }

    async def extract_enriched_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extracts both named entities and specialized concepts (tech terms, codes).
        Uses LLM for domain-specific concept extraction.
        """
        # 1. Base NER entities (Sync)
        entities = self.extract_entities(text)
        
        # 2. Specialized Concept Extraction via LLM (Async)
        from src.services.llm_service import LLMService
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt_template = ChatPromptTemplate.from_template(
            "Analyze this meeting transcript segment and extract key entities.\n"
            "Categories to extract:\n"
            "- technical_term (e.g. Docker, Python, RAG)\n"
            "- internal_code (e.g. project codes, employee IDs, specific FPT formats)\n"
            "- domain_word (specialized words related to the domain)\n\n"
            "Return JSON only: {{\"concepts\": [{{\"name\": \"term\", \"type\": \"technical_term|internal_code|domain_word\"}}]}}\n"
            "Transcript: {text}"
        )
        
        try:
            fast_llm = LLMService.get_fast_model()
            chain = prompt_template | fast_llm | JsonOutputParser()
            extractions = await chain.ainvoke({"text": text})
            concepts = extractions.get("concepts", [])
        except Exception as e:
            print(f"⚠️ Concept extraction failed: {e}")
            concepts = []

        return {
            "persons": list(entities["PER"]),
            "organizations": list(entities["ORG"]),
            "locations": list(entities["LOC"]),
            "concepts": concepts
        }

# Singleton instance
nlp_service = NLPService()
