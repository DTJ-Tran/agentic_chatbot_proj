import asyncio
import numpy as np
from qdrant_client import AsyncQdrantClient
from pathlib import Path
from fastembed import TextEmbedding
from typing import List, Dict, Any, Optional
from src.core.config import settings
from src.services.llm_service import LLMService
from src.tools.ot_mapper import OTTMapper

class VectorService:
    """
    Singleton service for Vector DB interaction and embedding generation.
    Supports eager loading (boost-up) to minimize first-request latency.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.cache_dir = str(Path(__file__).parent.parent.parent / "resources" / "models" / "fastembed_cache")
        # Switching to AsyncQdrantClient to prevent blocking the event loop
        self.qdrant = AsyncQdrantClient(url=settings.vector_db_url)
        self.collection = settings.vector_db_collection
        self.model = None # Initialized via warm_up()
        self.ot_mapper = None
        self._initialized = True

    def warm_up(self):
        """Eagerly initializes FastEmbed and trains the OTTMapper."""
        if self.model is None:
            print("📦 [VectorService] Loading local FastEmbed model (384D)...")
            self.model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5", 
                cache_dir=self.cache_dir
            )
            
            print("📦 [VectorService] Initializing OTT Projection map (384D -> 4096D)...")
            # 1. Define anchor sentences to learn the mathematical alignment
            anchors = [
                "What are the benefits for employees?",
                "How does FPT Software handle sick leave and vacation?",
                "Can you explain the performance review process in VNG?",
                "What is the policy for working from home?",
                "Are there any special allowances for senior engineers?",
                "How do FSoft policies compare against Viettel Digital?",
                "Tell me about maternity and paternity leave.",
                "What is the process for onboarding new hires?",
                "How do I request equipment or hardware upgrades?",
                "What forms are required for business travel reimbursement?"
            ]
            
            # 2. Get local 384D representations
            print("   -> Encoding anchors locally...")
            X_src = list(self.model.embed(anchors))
            
            # 3. Get API 4096D representations
            print("   -> Encoding anchors via API (ground truth)...")
            api_model = LLMService.get_embedding_model()
            Y_tgt = api_model.embed_documents(anchors)
            
            # 4. Train OT Mapper
            self.ot_mapper = OTTMapper()
            self.ot_mapper.train_alignment(X_src, Y_tgt)
            
            print("✅ [VectorService] Local embedding engine & OT map ready.")

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Performs a search against the vector database asynchronously."""
        # Ensure model is loaded (warm_up is sync but fast after first call)
        if self.model is None:
            self.warm_up()
        
        loop = asyncio.get_event_loop()
        
        try:
            # 1. Generate local dense embedding (384D)
            # Offload to executor because TextEmbedding.embed is a sync generator
            embeddings = await loop.run_in_executor(
                None, lambda: list(self.model.embed([query]))
            )
            local_vector = embeddings[0].tolist()
            
            # 2. Project 384D to 4096D using the pre-trained optimal transport map
            # This is a fast matrix multiplication, typically safe to run inline
            query_vector = self.ot_mapper.project(local_vector)

            # 3. Async Vector Search
            response = await self.qdrant.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit
            )

            return [
                {
                    "id": hit.id,
                    "content": hit.payload.get("text", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score
                }
                for hit in response.points
            ]
        except Exception as e:
            if settings.debug:
                print(f"❌ Async Vector search failed: {e}")
                import traceback
                traceback.print_exc()
            return []

    async def cleanup(self):
        """Clean up resources, including the Qdrant client."""
        if hasattr(self, 'qdrant'):
            print("Cleanup: closing Qdrant connection...")
            await self.qdrant.close()
