from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.engine.state import AgentState
from src.nodes.receptionist_node import receptionist_node
from src.nodes.react_worker_node import react_worker_node
from src.nodes.approval_node import approval_node
from src.nodes.meeting_note_node import meeting_node
from src.nodes.publishing_node import publishing_node
from src.nodes.scribing_node import scribing_node
"""
Some invariants:
    “Every interaction MUST produce at least one immutable raw event.”
    1 raw_envent = 1 conversation
    1 conversation = 1 cycle of graph-compilation -> 1 msg_id
       EXECUTION PATHS
   ┌──────┬──────┬──────┐
   ▼      ▼      ▼      ▼
 meeting  retrieval  publish  casual
   └──────┴──────┴──────┘
             ▼
         SCRIBING
             ▼
            END
"""
def create_rag_v2_graph():
    """
    Assembles the Production RAG v2 Multi-Agent Graph.
    Topology:
    receptionist -> casual/reject -> END
                  -> clarify -> END (wait user)
                  -> retrieval -> react_worker -> approval -> approved -> END
                                                           -> revise -> react_worker
                  -> meeting -> meeting_node -> END
    receptionist -> ...
                  -> publishing -> scribing -> publishing -> END
                  -> meeting -> meeting_node -> scribing -> publishing -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("receptionist", receptionist_node)
    workflow.add_node("react_worker", react_worker_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("meeting_node", meeting_node)
    workflow.add_node("scribing", scribing_node)
    workflow.add_node("publishing", publishing_node)

    # 2. Define Conditional Edges for Receptionist
    workflow.add_conditional_edges(
        "receptionist",
        lambda state: state.get("route", "retrieval"),
        {
            "casual": "scribing", # already done - log immediately
            "reject": END,
            "halting": END,
            "retrieval": "react_worker",
            "meeting": "meeting_node",
            "publishing": "publishing"
        }
    )

    # 3. Define Flow Edges
    workflow.add_edge("react_worker", "approval")
    workflow.add_edge("meeting_node", "scribing")
    # workflow.add_edge("scribing", "publishing")
    workflow.add_edge("publishing", "scribing")
    # 4. Define Approval edges
    workflow.add_conditional_edges(
        "approval",
        lambda state: state.get("route", "approved"),
        {
            "approved": "scribing",
            "revise": "react_worker"
        }
    )
    # Scribing always terminal
    workflow.add_edge("scribing", END)
    workflow.set_entry_point("receptionist")

    # 5. Compile with Memory for HITL persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
