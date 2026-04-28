import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import onnxruntime as ort
from transformers import AutoProcessor
import librosa
from src.core.config import settings

class ASRService:
    """
    Service for Automatic Speech Recognition using Moonshine ONNX models.
    Torch-free implementation for production stability.
    """
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or settings.asr_model_path)
        self.encoder_path = self.model_dir / "encoder_model.onnx"
        self.decoder_path = self.model_dir / "decoder_model.onnx"
        
        if not self.encoder_path.exists() or not self.decoder_path.exists():
            # Try alternate names if needed
            self.encoder_path = self.model_dir / "encoder.onnx"
            self.decoder_path = self.model_dir / "decoder.onnx"

        print(f"🚀 Initializing ASRService with model at {self.model_dir}")
        self.encoder_session = ort.InferenceSession(str(self.encoder_path))
        self.decoder_session = ort.InferenceSession(str(self.decoder_path))
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        except Exception:
            self.processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
            
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        self._detect_dimensions()

    def _detect_dimensions(self):
        # Default Tiny
        self.num_layers = 6
        self.num_heads = 8
        self.head_dim = 36
        
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.num_layers = config.get("decoder_num_hidden_layers", config.get("num_hidden_layers", 6))
            self.num_heads = config.get("decoder_num_attention_heads", config.get("num_attention_heads", 8))
            hidden_size = config.get("hidden_size", 288)
            self.head_dim = hidden_size // self.num_heads

    def preprocess(self, audio_data: np.ndarray, target_sr: int = 16000) -> np.ndarray:
        # Normalize audio for Moonshine
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0.001:
            audio_data = audio_data * (0.075 / rms)
        return np.clip(audio_data, -1.0, 1.0)

    def transcribe(self, audio_path_or_array: Union[str, Path, np.ndarray]) -> str:
        if isinstance(audio_path_or_array, (str, Path)):
            # Use librosa for loading and automatic resampling to 16kHz mono
            audio_array, _ = librosa.load(str(audio_path_or_array), sr=16000, mono=True)
        else:
            audio_array = audio_path_or_array

        audio_array = self.preprocess(audio_array)
        
        # Encode
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="np")
        enc_inputs = {inp.name: inputs.input_values for inp in self.encoder_session.get_inputs()}
        encoder_hidden_states = self.encoder_session.run(None, enc_inputs)[0]
        
        # Decode
        return self._decode_greedy(encoder_hidden_states)

    def _decode_greedy(self, encoder_hidden_states: np.ndarray, max_new_tokens: int = 150) -> Dict[str, Any]:
        start_time = time.time()
        B, T_enc, _ = encoder_hidden_states.shape
        decoder_input_ids = np.array([[self.bos_token_id]], dtype=np.int64)
        generated_tokens = []
        token_probs = []
        
        past_kv = {}
        use_cache = False
        
        for i in range(self.num_layers):
            past_kv[f"past_key_values.{i}.decoder.key"] = np.zeros((B, self.num_heads, 0, self.head_dim), dtype=np.float32)
            past_kv[f"past_key_values.{i}.decoder.value"] = np.zeros((B, self.num_heads, 0, self.head_dim), dtype=np.float32)
            past_kv[f"past_key_values.{i}.encoder.key"] = np.zeros((B, self.num_heads, T_enc, self.head_dim), dtype=np.float32)
            past_kv[f"past_key_values.{i}.encoder.value"] = np.zeros((B, self.num_heads, T_enc, self.head_dim), dtype=np.float32)

        decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        
        encoder_kv_set = False

        for _ in range(max_new_tokens):
            inputs = {}
            for name in decoder_input_names:
                if 'input_ids' in name: inputs[name] = decoder_input_ids
                elif 'encoder_hidden_states' in name: inputs[name] = encoder_hidden_states
                elif 'encoder_attention_mask' in name: inputs[name] = np.ones((B, T_enc), dtype=np.int64)
            
            inputs["use_cache_branch"] = np.array([use_cache], dtype=bool)
            inputs.update(past_kv)
            
            outputs = self.decoder_session.run(None, inputs)
            logits = outputs[0]
            kv_outputs = outputs[1:]
            
            # Update KV
            new_past_kv = {}
            idx = 0
            for i in range(self.num_layers):
                new_past_kv[f"past_key_values.{i}.decoder.key"] = kv_outputs[idx]; idx += 1
                new_past_kv[f"past_key_values.{i}.decoder.value"] = kv_outputs[idx]; idx += 1
                if not encoder_kv_set:
                    new_past_kv[f"past_key_values.{i}.encoder.key"] = kv_outputs[idx]; idx += 1
                    new_past_kv[f"past_key_values.{i}.encoder.value"] = kv_outputs[idx]; idx += 1
                else:
                    idx += 2 # Skip encoder KV as they are constant
                    new_past_kv[f"past_key_values.{i}.encoder.key"] = past_kv[f"past_key_values.{i}.encoder.key"]
                    new_past_kv[f"past_key_values.{i}.encoder.value"] = past_kv[f"past_key_values.{i}.encoder.value"]
            
            past_kv = new_past_kv
            encoder_kv_set = True
            
            next_token_logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            exp_logits = np.exp(next_token_logits - np.max(next_token_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            next_token = np.argmax(probs, axis=-1)
            token_prob = probs[0, next_token[0]]
            
            if next_token[0] == self.eos_token_id:
                break
            
            generated_tokens.append(next_token[0])
            token_probs.append(token_prob)
            
            decoder_input_ids = next_token.reshape(1, 1)
            use_cache = True
            
        transcript = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        latency_ms = int((time.time() - start_time) * 1000)
        
        confidence = np.mean(token_probs) if token_probs else 0.0
        
        return {
            "text": transcript,
            "latency_ms": latency_ms,
            "confidence": float(confidence),
            "engine": "Moonshine-Base"
        }

# Singleton instance
asr_service = ASRService()
