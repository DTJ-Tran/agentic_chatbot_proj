from typing import Dict, Any

class MemRaidController:
    """
    Decides whether to store informaton in the persistent memory RAID.
    Helps filter noise and keeps 'SYNTHESIS' payloads high-quality.
    """
    IMPORTANCE_THRESHOLD = 6

    @staticmethod
    def evaluate(intent_data: Dict[str, Any]) -> str:
        """
        Determines the memory action based on intent results.
        Returns:
            {
            "store_memory": bool,
            }
        """
        score = intent_data.get("importance_score", 0)
        category = intent_data.get("category", "casual")
        export_signal = intent_data.get("export_signal", False)


        # Memory Decision (Independent) - default for casual conv / low important score
        store_memory = False

        # 2. High value technical/meeting info is stored
        # Store retrieval / meeting 
        # or store really important conversation based on LLM-judgement
        if category in ["retrieval", "meeting"] or score >= MemRaidController.IMPORTANCE_THRESHOLD:
           store_memory = True

        # 3. Export requests need conversation history available for publishing
        if export_signal:
            store_memory = True


        return {
            "store_memory" : store_memory,
        }
