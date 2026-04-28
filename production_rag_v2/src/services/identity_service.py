import onnxruntime as ort
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from typing import Dict, List, Optional, Tuple, Any
import os
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.core.config import settings

class SpeakerIdentityService:
    """
    Persistent Speaker Identity service using WESpeaker ECAPA-TDNN ONNX model.
    Uses Qdrant for semantic caching of speaker 'voice prints' (embeddings).
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Speaker embedding model not found at {model_path}")
        
        # Initialize ONNX session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Qdrant configuration
        self.qdrant = AsyncQdrantClient(url=settings.vector_db_url)
        self.collection_name = settings.speaker_collection
        self.threshold = 0.75  # Cosine similarity threshold
        self.vector_dim = 192  # Based on WESpeaker ONNX output shape for this specific model
        self._initialized = False

    async def init_collection(self):
        """Ensure the speaker identity collection exists in Qdrant."""
        if self._initialized:
            return
            
        collections = await self.qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name not in collection_names:
            print(f"📦 Creating speaker identity collection: {self.collection_name} ({self.vector_dim}D)")
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
        self._initialized = True

    def _extract_fbank(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract Kaldi-style Fbank features."""
        # Ensure 16kHz
        if sr != 16000:
            pass # Implementation assumes 16k

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Hyperparameters from Wespeaker config.yaml
        feats = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=1.0,
            sample_frequency=sr,
            window_type='hamming'
        )
        
        # Segment-level CMVN
        feats = feats - torch.mean(feats, dim=0, keepdim=True)
        return feats.unsqueeze(0).numpy() # [1, T, 80]

    def get_embedding(self, audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """Generate a persistent embedding for an audio clip."""
        if len(audio) < 1600: # Less than 0.1s
            return None
            
        feats = self._extract_fbank(audio, sr)
        # onnx input name is 'feats' for ECAPA
        outputs = self.session.run(None, {'feats': feats})
        embedding = outputs[0][0] # First batch, first result
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    async def match_speaker(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Match an audio clip against the persistent Qdrant speaker registry."""
        if not self._initialized:
            await self.init_collection()

        embedding = self.get_embedding(audio, sr)
        if embedding is None:
            return "UNKNOWN"

        # Search Qdrant for the closest match using the new Query API
        search_result = await self.qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=1,
            score_threshold=self.threshold
        )

        if search_result.points:
            best_match = search_result.points[0]
            speaker_id = best_match.payload.get("speaker_id")
            print(f"🎯 Identity Match FOUND (Qdrant): {speaker_id} (score: {best_match.score:.4f})")
            return speaker_id
        else:
            # Create a new speaker identity
            new_id = f"Speaker_{uuid.uuid4().hex[:6]}"
            print(f"🆕 Identity NEW Speaker (Qdrant): {new_id}")
            
            # Upsert into Qdrant
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={"speaker_id": new_id, "source": "diarization_flow"}
                    )
                ]
            )
            return new_id

# Lazy initialization
_identity_service = None

def get_identity_service():
    global _identity_service
    if _identity_service is None:
        _identity_service = SpeakerIdentityService(settings.speaker_model_path)
    return _identity_service
