import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Force JAX to use CPU to avoid TPU errors on Mac
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Path resolution for unified resources
V2_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = V2_ROOT / "resources"

class Settings(BaseSettings):
    load_dotenv()
    
    model_config = SettingsConfigDict(
        env_file=str(V2_ROOT.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # FIREWORKS
    fireworks_api_key: str = Field(..., env="FIREWORKS_API_KEY")
    fireworks_chat_model: str = Field(
        "accounts/fireworks/models/gpt-oss-20b", 
        env="FIREWORKS_CHAT_MODEL"
    )
    fireworks_embedding_model: str = Field(
        "accounts/fireworks/models/qwen3-embedding-8b", 
        env="FIREWORKS_EMBEDDING_MODEL"
    )

    # DOMAIN
    server_domain: str = Field("localhost", validation_alias="SERVER_DOMAIN")

    # QDRANT / VECTOR DB
    vector_db_url: str = Field(..., validation_alias="VECTOR_DB_URL")
    vector_db_collection: str = Field("fpt_policy", validation_alias="_DEFAULT_VECTOR_COLLECTION")
    emb_size: int = Field(4096, validation_alias="EMB_SIZE")

    # REDIS
    redis_url: str = Field(..., validation_alias="UPSTASH_REDIS_REST_URL")
    redis_token: str = Field(..., validation_alias="UPSTASH_REDIS_REST_TOKEN")

    # LANGFUSE
    langfuse_public_key: str = Field(..., validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(..., validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL")

    # TAVILY
    tavily_api_key: str = os.getenv("TAVILY_API") #Field(..., validation_alias="TAVILY_API")
    
    # MONGO DB
    mongo_url: str = Field("mongodb://localhost:27017/", validation_alias="MONGO_DB_CONNECTION_STR")
    mongo_db_name: str = Field("FPT_Internship_DocDB", validation_alias="MONGO_DB_DATABASE_NAME")

    # NEO4J
    neo4j_url: str = Field("bolt://localhost:7687", validation_alias="NE_4_J_URL")
    neo4j_user: str = Field("neo4j", validation_alias="NEO_4_J_USR_NAME")
    neo4j_password: str = Field("password", validation_alias="NEO_4_J_PASSWORD")

    # APP SETTINGS (Local Resources)
    debug: bool = False
    log_level: str = "INFO"
    use_edge_llm: bool = False
    
    # Unified Resource Paths
    asr_model_path: str = Field(
        str(RESOURCES_DIR / "asr/moonshine_model/onnx/merged/tiny-vi/float"), 
        env="ASR_MODEL_PATH"
    )
    vncorenlp_path: str = Field(
        str(RESOURCES_DIR / "VnCoreNLP"), 
        env="VNCORENLP_PATH"
    )
    fallback_asr_model_path: str = Field(
        str(RESOURCES_DIR / "asr/wav2Vec-int-8-ASR"), 
        env="FALLBACK_ASR_MODEL_PATH"
    )
    kenlm_vn_path: str = Field(
        str(RESOURCES_DIR / "asr/vietnamese.bin"), 
        env="KENLM_VN_PATH"
    )
    kenlm_en_path: str = Field(
        str(RESOURCES_DIR / "asr/english.bin"), 
        env="KENLM_EN_PATH"
    )
    
    # New Unified Resource Definitions
    vad_model_path: str = Field(
        str(RESOURCES_DIR / "silero_VAD/model.onnx"),
        env="VAD_MODEL_PATH"
    )
    speaker_model_path: str = Field(
        str(RESOURCES_DIR / "sv/ecapa-tdnn-sv-512/voxceleb_ECAPA512_LM.onnx"),
        env="SPEAKER_MODEL_PATH"
    )
    qwen_align_model_path: str = Field(
        str(RESOURCES_DIR / "qwen-3-force_alignner-0_6B_onnix"),
        env="QWEN_ALIGN_MODEL_PATH"
    )
    edge_llm_root: str = Field(
        str(RESOURCES_DIR / "llm_edge"),
        env="EDGE_LLM_ROOT"
    )
    
    # Diarization & ASR Extras
    pyannot_api: Optional[str] = Field(None, validation_alias="PYANNOT_API")
    hf_token: Optional[str] = Field(None, validation_alias="HF_TOKEN")
    speaker_collection: str = Field("speaker_identity", validation_alias="SPEAKER_COLLECTION")

    # Thresholds
    snr_threshold_db: float = 15.0
    asr_confidence_threshold: float = 0.7
    
settings = Settings()
