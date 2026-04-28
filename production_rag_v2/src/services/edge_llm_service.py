import os
from langchain_community.llms.llamacpp import LlamaCpp
from src.core.config import settings

class EdgeLLMService:
    """
    Service for loading and running smaller 'edge' model locally using llama-cpp-python.
    Decision (Qwen-2.5-3B-Instruct-Q5) - For Determine whether to use ReAct Agent / just basic response 
    """
    _receptionist_model = None

    def __get_model_path(self, model_name: str):
        # Point to the unified resources in v2
        return os.path.join(settings.edge_llm_root, model_name)

    @classmethod
    def get_decision_model(cls):
        """Single model for policy decision and response routing."""
        if cls._receptionist_model is None:
            instance = cls()
            model_path = instance.__get_model_path("qwen3_0.6B_Q8_0.gguf")
            print(f"📦 [EdgeLLM] Loading Senior Receptionist: {os.path.basename(model_path)} ...")
            
            cls._receptionist_model = LlamaCpp(
                model_path=model_path,
                n_ctx=4096,
                n_batch=512,
                n_threads=4,
                f16_kv=True,
                temperature=0.3,
                n_gpu_layers=-1, 
                verbose=False,
                top_p=0.95,
                stop=["</RESPONSE>"],
                max_tokens=600,
                logits_all=False,
                model_kwargs={
                    "min_p": 0.05,
                    "presence_penalty": 1.1
                }
            )
            
        return cls._receptionist_model

    @classmethod
    def get_edge_model(cls):
        """Legacy compatibility - defaults to decision model."""
        return cls.get_decision_model()

    @classmethod
    def cleanup(cls):
        """Release GPU resources by clearing model references."""
        print("🧹 [EdgeLLM] Releasing GPU resources...")
        cls._receptionist_model = None
        import gc
        gc.collect()
