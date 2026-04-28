import os
import time
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiCorrectionService:
    def __init__(self, model_name: str = "models/gemini-3.1-flash-lite-preview"):
        # 1. Try standard load_dotenv (current working directory)
        load_dotenv()
        
        # 2. Fallback: Try project root specifically
        if not os.getenv("GOOGLE_API_KEY"):
            root_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"
            load_dotenv(dotenv_path=root_env)
            
        # 3. Fallback: Try sibling production_rag_v2 root
        if not os.getenv("GOOGLE_API_KEY"):
            v2_env = Path(__file__).resolve().parent.parent.parent / ".env"
            load_dotenv(dotenv_path=v2_env)
            
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or .env file.")
        
        # Ensure it's in os.environ for genai.upload_file
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"🚀 GeminiCorrectionService initialized with model: {model_name}")

    def correct_audio_transcript(self, audio_path: str, wav2vec_ipa: str, 
                                 moonshine_text: str, language_hint: str, 
                                 reason: str) -> Optional[str]:
        """
        Uploads audio and uses Gemini's native audio multi-modality to refine the transcript.
        """
        try:
            # 1. Upload the audio file
            print(f"☁️ Uploading audio to Gemini: {audio_path}")
            audio_file = genai.upload_file(path=audio_path)
            
            # Wait for processing
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise ValueError(f"Audio file processing failed: {audio_file.name}")

            # 2. Prepare the prompt
            prompt = f"""
            You are an expert speech transcriber and phonetic analyst. 
            I am providing you with an audio clip and two ASR candidates.
            
            Target Language: {language_hint}
            Reason for LLM intervention: {reason}
            
            ASR Candidates:
            - Moonshine (Grapheme Draft): "{moonshine_text}"
            - Wav2Vec (IPA Phoneme Draft): "{wav2vec_ipa}"
            
            Task:
            Listen to the audio and use the IPA Phoneme Draft as a high-fidelity acoustic hint. 
            The IPA draft may contain the exact sounds but might be spelled phonetically. 
            The Moonshine draft might have the correct structure but miss specific sounds or jargon.
            
            Provide the most accurate and coherent transcript in {language_hint}. 
            Respond ONLY with the corrected text, no preamble or explanation.
            """

            # 3. Generate content
            response = self.model.generate_content([audio_file, prompt])
            
            # 4. Cleanup (optional but good practice)
            genai.delete_file(audio_file.name)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ Gemini Correction Error: {e}")
            return None

# Singleton
try:
    llm_correction_service = GeminiCorrectionService()
except Exception as e:
    print(f"⚠️ Could not initialize GeminiCorrectionService: {e}")
    llm_correction_service = None
