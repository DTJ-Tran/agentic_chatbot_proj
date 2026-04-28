import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from src.core.config import settings
from src.services.llm_service import LLMService
from src.tools.lda_summarizer import LDASummarizer
from src.tools.bm25_scorer import BM25Scorer
from src.utils.text_norm import full_clean, detect_language
from src.tools.vn_core_wrapper import VnCoreNLPWrapper
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_template(
    "Analyze the user query and provide 3 refined search queries (What/Why/How).\n"
    "Return JSON: {{\"search_queries\": [\"q1\", \"q2\", \"q3\"], \"time_range\": \"months\"}}\n"
    "User Query: {query}"
)

class SearchService:
    """
    Production-grade Search Orchestrator.
    Handles expansion, multi-search, ML-based cleansing, and ranking.
    """
    
    def __init__(self):
        self.tavily = TavilyClient(api_key=settings.tavily_api_key)
        self.lda = LDASummarizer()
        self.vn_core = VnCoreNLPWrapper()
        self.logger = logging.getLogger(__name__)

    async def _expand_query(self, query: str) -> Dict[str, Any]:
        """Use a fast LLM to expand the query into multiple search perspectives."""
        try:
            llm = LLMService.get_fast_model()
            chain = QUERY_EXPANSION_PROMPT | llm | JsonOutputParser()
            expansion = await chain.ainvoke({"query": query})
            if not expansion:
                return {"search_queries": [query], "time_range": "months"}
            return expansion
        except Exception as e:
            self.logger.warning("Query expansion failed: %s", e)
            return {"search_queries": [query], "time_range": "months"}

    async def _safe_search(self, q: str) -> List[Dict[str, Any]]:
        """Perform a single search safely in a thread."""
        try:
            response = await asyncio.to_thread(
                self.tavily.search,
                query=q,
                search_depth="basic",
                max_results=5,
                include_raw_content=True
            )
            return response.get("results", [])
        except Exception as e:
            msg = str(e)
            if "Forbidden" in msg or "quota" in msg or "429" in msg:
                self.logger.warning("Tavily quota exceeded for query: %s", q)
            else:
                self.logger.warning("Tavily error: %s", e)
            return []

    async def _process_document(self, doc: Dict[str, Any], original_query: str) -> Optional[Dict[str, Any]]:
        """Clean, segment, topic-model, and score a single search result."""
        content = doc.get("raw_content") or doc.get("content") or ""
        if not content:
            return None
            
        # 1. Cleaning
        cleaned_text = full_clean(content)
        if len(cleaned_text) < 100:
            return None
            
        # 2. Language & Segmentation
        lang = detect_language(cleaned_text)
        if lang == "vi":
            segmented_text = self.vn_core.segment(cleaned_text)
        else:
            segmented_text = cleaned_text
            
        sentences = [s.strip() for s in segmented_text.split('.') if len(s.strip()) > 20]
        
        # 3. Topic Modeling (LDA)
        relevant_sentences = self.lda.summarize(
            text=segmented_text,
            sentences=sentences,
            query_hints=[original_query],
            use_vn=(lang == "vi")
        )
        
        # Fallback: if LDA returns nothing, use first few sentences as snippets
        if not relevant_sentences:
            relevant_sentences = sentences[:5] if sentences else [cleaned_text[:500]]

        # 4. Lexical Scoring (BM25)
        quality = 0.0
        top_idx = list(range(min(5, len(relevant_sentences))))
        try:
            scorer = BM25Scorer([s.split() for s in relevant_sentences])
            scores = scorer.get_scores(original_query)
            bm25_quality, bm25_top_idx = scorer.compute_doc_quality(scores)
            if bm25_quality > 0:
                quality = bm25_quality
                top_idx = bm25_top_idx.tolist() if hasattr(bm25_top_idx, 'tolist') else list(bm25_top_idx)
        except Exception:
            pass  # BM25 failure is not fatal

        # Fallback: use Tavily's own relevance score when BM25 degenerates
        # This handles specific internal terminology that BM25 can't score
        tavily_score = float(doc.get("score", 0.0))
        if quality <= 0 and tavily_score >= 0.3:
            quality = tavily_score * 0.5  # Discount external score slightly
            
        if quality <= 0:
            return None
            
        return {
            "url": doc.get("url"),
            "title": doc.get("title"),
            "score": quality,
            "snippets": [relevant_sentences[i] for i in top_idx if i < len(relevant_sentences)],
            "full_content": cleaned_text[:2000]  # Cap for context safety
        }

    async def run_pipeline(self, query: str, max_docs: int = 5) -> List[Dict[str, Any]]:
        """Main entry point for the search research pipeline."""
        start_time = time.time()
        
        # Phase 1: Expansion
        expansion = await self._expand_query(query)
        queries = expansion.get("search_queries", [query])
        if settings.debug:
            self.logger.info("[Search] Phase 1 expanded into %s queries: %s", len(queries), queries)
        
        # Phase 2: Parallel Web Search
        search_tasks = [self._safe_search(q) for q in queries]
        search_results_nested = await asyncio.gather(*search_tasks)
        
        # Flat list & Deduplicate
        seen_urls = set()
        flat_results = []
        for sublist in search_results_nested:
            if not sublist:
                continue
            for res in sublist:
                if res and res.get("url") and res["url"] not in seen_urls:
                    seen_urls.add(res["url"])
                    flat_results.append(res)
        
        if settings.debug:
            self.logger.info("[Search] Phase 2 Tavily returned %s raw unique results.", len(flat_results))
        if not flat_results:
            self.logger.warning("[Search] No web results from Tavily.")
            return []
            
        # Phase 3: Parallel Processing (LDA/Scoring)
        proc_tasks = [self._process_document(res, query) for res in flat_results]
        processed_docs = await asyncio.gather(*proc_tasks)
        
        # Phase 4: Final Ranking
        final_list = [d for d in processed_docs if d]
        final_list.sort(key=lambda x: x["score"], reverse=True)
        
        discarded = len(flat_results) - len(final_list)
        if settings.debug:
            self.logger.info(
                "[Search] Phase 3/4 %s passed quality filter (%s discarded).",
                len(final_list),
                discarded,
            )
            self.logger.info(
                "[Search] Pipeline finished in %.1fs. Docs: %s",
                time.time() - start_time,
                len(final_list),
            )
        return final_list[:max_docs]
