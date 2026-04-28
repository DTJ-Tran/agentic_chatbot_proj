import json
from typing import Dict, Any
from src.services.llm_service import llm_service

INTENT_PROMPT = """You are the Intent Classification Module.
Analyze the user message and determine the context and goals.

MESSAGE: {message}

CATEGORIES:
- casual: General chat, greetings, personality checks.
- retrieval: Asking questions that require searching knowledge (e.g., "What is...", "How do I...").
- meeting: Transcribing or summarizing discussions between people.
- export: Explicit request to save, export, or log information to Notion/External.

Return a JSON object:
{{
  "category": "casual" | "retrieval" | "meeting" | "export",
  "importance_score": 0-10,
  "export_signal": true | false,
  "export_mode": "RAW" | "SYNTHESIS",
  "reasoning": "short explanation"
}}

GUIDELINES:
- If user says "Export this", "Save to Notion", or "Log this", export_signal is TRUE and mode is RAW (Literal Transcript).
- If user says "Summarize", "Synthesize", "Refine", or "Compress this", export_signal is TRUE and mode is SYNTHESIS (Paraphrased/Concise).
- High importance ( > 7 ) should be assigned to complex technical answers or verified facts.
"""


class IntentModule:
    
    @staticmethod
    async def classify(message: str) -> Dict[str, Any]:
        """Classifies the user intent and provides signals for memory/export."""
        if not message or not message.strip():
            return {
                "category": "casual",
                "importance_score": 0,
                "export_signal": False,
                "export_mode": "SYNTHESIS",
                "reasoning": "Empty message"
            }

        prompt = INTENT_PROMPT.format(message=message)
        # Use the fast model for latency-sensitive classification
        raw_res = await llm_service.generate(prompt)
        
        try: # if in here export signal is True
            # Basic JSON extraction
            clean_res = raw_res.strip()
            if "```json" in clean_res:
                clean_res = clean_res.split("```json")[-1].split("```")[0]
            elif "```" in clean_res:
                clean_res = clean_res.split("```")[-1].split("```")[0]
            
            result = json.loads(clean_res) # handle the case if the export_signal is not a bool but a str
            result["export_signal"] = str(result.get("export_signal", "false")).lower() == "true"
            result["importance_score"] = int(result.get("importance_score", 0))
            result["category"] = str(result.get("category", "casual")).lower().strip()
            result["export_mode"] = str(result.get("export_mode", "SYNTHESIS")).upper().strip()

            # Deterministic normalization: export category must behave like export.
            if result["category"] == "export":
                result["export_signal"] = True

            # Heuristic guardrail for common export verbs when the model is inconsistent.
            msg_lower = message.lower()
            if any(k in msg_lower for k in ["export", "save to notion", "send to notion", "log this", "publish"]):
                result["export_signal"] = True
                if any(k in msg_lower for k in ["summarize", "synthesize", "refine", "compress"]):
                    result["export_mode"] = "SYNTHESIS"
                else:
                    result["export_mode"] = "RAW"
            return result
        
        except Exception:
            return {
                "category": "retrieval",
                "importance_score": 5,
                "export_signal": False,
                "export_mode": "SYNTHESIS",
                "reasoning": "Fallback classification due to parse error"
            }
