import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Any, Optional

class BM25Scorer:
    """
    Handles lexical scoring using BM25 and document-level relevance calculation 
    based on information entropy and signal density.
    """
    
    def __init__(self, tokenized_docs: Optional[List[List[str]]] = None):
        self.bm25 = None
        if tokenized_docs:
            self.bm25 = BM25Okapi(tokenized_docs)

    def fit(self, tokenized_docs: List[List[str]]):
        """Initialize the BM25 index with tokenized documents."""
        self.bm25 = BM25Okapi(tokenized_docs)

    def get_scores(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for a query against all indexed documents."""
        if not self.bm25:
            # Lazy init return 0
            return np.zeros(1)
        query_tokens = query.lower().split()
        return np.array(self.bm25.get_scores(query_tokens))

    def compute_doc_quality(self, scores: np.ndarray, top_k: int = 5) -> Tuple[float, np.ndarray]:
        """
        Calculates a document quality score based on relevance mass and information entropy.
        """
        if len(scores) == 0:
            return 0.0, np.array([])
            
        total_mass = scores.sum()
        if total_mass <= 0:
            return 0.0, np.array([])

        p = scores / total_mass
        n = len(scores)
        entropy = -np.sum(p * np.log(p + 1e-12))
        
        norm_entropy = entropy / np.log(n) if n > 1 else 1.0
        norm_entropy = max(0.0, norm_entropy)

        target_k = min(top_k, n)
        topk_idx = np.argpartition(scores, -target_k)[-target_k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        
        relevance_density = scores[topk_idx].sum() / total_mass
        quality_score = relevance_density * norm_entropy
        
        return float(quality_score), topk_idx
