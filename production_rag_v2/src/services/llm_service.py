from langchain_fireworks import ChatFireworks
from src.core.config import settings

class LLMService:
    """
    Centralized provider for LLM and Embedding models.
    Ensures persistent clients and systematic access to avoid session leaks.
    """
    _chat_model = None
    _fast_model = None
    _embedding_model = None

    @classmethod
    def define_chat_model(cls, model: str, temp=0.5):
        cls._chat_model = ChatFireworks(
            api_key=settings.fireworks_api_key,
            model=model,
            temperature=temp
        )
        return cls._chat_model

    @classmethod
    def get_chat_model(cls) -> ChatFireworks:
        if cls._chat_model is None:
            cls._chat_model = ChatFireworks(
                api_key=settings.fireworks_api_key,
                model=settings.fireworks_chat_model,
                temperature=0.2
            )
        return cls._chat_model

    @classmethod
    def get_fast_model(cls) -> ChatFireworks:
        """Lightweight model for faster/cheaper reasoning like query expansion."""
        if cls._fast_model is None:
            cls._fast_model = ChatFireworks(
                api_key=settings.fireworks_api_key,
                model="accounts/fireworks/models/gpt-oss-20b",
                temperature=0.1
            )
        return cls._fast_model

    @classmethod
    def get_embedding_model(cls):
        """Returns the fireworks embedding model to match the 4096d Qdrant collection."""
        if cls._embedding_model is None:
            from langchain_fireworks import FireworksEmbeddings
            cls._embedding_model = FireworksEmbeddings(
                api_key=settings.fireworks_api_key,
                model=settings.fireworks_embedding_model
            )
        return cls._embedding_model

    async def generate(self, prompt: str) -> str:
        """Wrapper for generating text from the chat model."""
        model = self.get_chat_model()
        try:
            response = await model.ainvoke(prompt)
        except RuntimeError as e:
            if "Session is closed" not in str(e):
                raise
            # Recreate the shared Fireworks client after an externally closed aiohttp session.
            self.__class__._chat_model = None
            model = self.get_chat_model()
            response = await model.ainvoke(prompt)
        return response.content

    @classmethod
    def cleanup(cls):
        """Release resources and clear singletons."""
        if cls._chat_model or cls._fast_model or cls._embedding_model:
            print("🧹 [LLMService] Releasing resources...")
            cls._chat_model = None
            cls._fast_model = None
            cls._embedding_model = None
            import gc
            gc.collect()

# Singleton instance
llm_service = LLMService()
