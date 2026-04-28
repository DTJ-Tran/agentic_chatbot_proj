import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Optional, Tuple
import os
from pathlib import Path

class LDASummarizer:
    def __init__(self, resources_path: Optional[str] = None):
        self._vn_stopwords = self._load_vn_stopwords(resources_path)
        self._en_vectorizer = CountVectorizer(stop_words="english", analyzer='word')
        self._vn_vectorizer = CountVectorizer(
            stop_words=list(self._vn_stopwords),
            token_pattern=r"(?u)\b\w+\b",
            analyzer='word'
        )

    def _load_vn_stopwords(self, resources_path: Optional[str]) -> set:
        """Load Vietnamese stopwords from file."""
        if resources_path is None:
            # Default to the sibling resources directory
            resources_path = Path(__file__).parent.parent.parent / "resources" / "vn_stopwords.txt"
        
        try:
            with open(resources_path, "r", encoding="utf-8") as f:
                return {line.strip().lower() for line in f if line.strip()}
        except Exception as e:
            print(f"Warning: Could not load VN stopwords from {resources_path}: {e}")
            return set()

    def summarize(
        self, 
        text: str, 
        sentences: List[str],
        num_topics: int = 5, 
        top_k: int = 10, 
        use_vn: bool = True,
        query_hints: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract the most representative sentences using LDA and query guidance.
        """
        if not sentences or len(sentences) <= top_k:
            return sentences

        vectorizer = self._vn_vectorizer if use_vn else self._en_vectorizer
        
        try:
            # Transform sentences into document-term matrix
            X = vectorizer.fit_transform(sentences)
            
            # Run LDA to find latent topics
            lda = LatentDirichletAllocation(
                n_components=min(num_topics, len(sentences)),
                random_state=42
            )
            topic_dist = lda.fit_transform(X)
            
            # Determine target topic (highest overall mean or query-aligned)
            target_topic = np.argmax(topic_dist.mean(axis=0))
            
            if query_hints:
                query_topic_scores = []
                for q in query_hints:
                    if not q.strip(): continue
                    try:
                        q_vec = vectorizer.transform([q])
                        q_topic_dist = lda.transform(q_vec)
                        query_topic_scores.append(q_topic_dist.mean(axis=0))
                    except:
                        continue
                
                if query_topic_scores:
                    avg_query_topic = np.vstack(query_topic_scores).mean(axis=0)
                    target_topic = np.argmax(avg_query_topic)

            # Sort sentences by their alignment with the target topic
            target_scores = topic_dist[:, target_topic]
            ranked_indices = np.argsort(target_scores)[::-1]
            
            return [sentences[i].strip() for i in ranked_indices[:top_k]]
            
        except Exception as e:
            print(f"LDA Error: {e}")
            return sentences[:top_k]
