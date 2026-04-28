from typing import Dict, Any
from langgraph.types import interrupt
from src.engine.state import AgentState

async def approval_node(state: AgentState) -> Dict[str, Any]:
    """
    Human-in-the-loop checkpoint for V2.
    """
    msg = state.get("msg", {})
    
    reply = interrupt({
        "question": (
            "✅ Press Enter to approve and start a new conversation.\n"
            "✏️  Or provide feedback to refine the answer."
        )
    })

    user_input = str(reply).strip()
    
    if not user_input or user_input.lower() in ("yes", "y"):
        return {"route": "approved"}

    # Revision path
    msg["msg_body"]["human_feedback"] = user_input
    return {"msg": msg, "route": "revise"}
