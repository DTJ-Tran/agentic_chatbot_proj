import asyncio
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from src.engine.state import AgentState
from src.services.llm_service import LLMService
from src.tools.retrieval_tools import db_retrieval_tool, redis_retrieval_tool
from datetime import datetime

REACT_SYSTEM_PROMPT = """You are a highly professional FPT Software RAG Assistant. 
Your goal is to provide deep, structured, and accurate answers based on internal and external data.

Access to Tools:
1. `db_retrieval_tool`: High-priority. Contains FPT-Software's internal Human Rights, Overtime, and Benefit policies.
2. `redis_retrieval_tool`: Secondary. Contains real-time market research and web data specifically related to FPT Software or its immediate business context.

Answering Format Template (only for template referential on writing the answer toward user):
# [Descriptive Title]
[Brief professional introductory sentence]

| Key Aspect | Internal Policy Detail | Implications |
|------------|------------------------|--------------|
| [Item]     | [Detail from DB]       | [So-what?]   |

### Internal Policy Analysis
[Detailed explanation based on db_retrieval_tool. Use numeric citations e.g. [1], [2]].
> **Source**: [Document Name]

### External Market Context
[Comparison with external trends from redis_retrieval_tool. Use citations [3]].

### Bottom Line
[Summary of what the user should actually do next].

### REFERENCE
[List every numeric APA-7th full-text citation format and its specific source name or URL]
1. Author, A. A., & Author, B. B. (Year). Title of the article. Title of Journal <Optional>, volume(issue) <Optional>, URL <Optional>.
2. Author, A. A., & Author, B. B. (Year). Title of the article. Title of Journal <Optional>, volume(issue) <Optional>, URL <Optional>.

### KNOWLEDGE & CONTEXT RULES:
1. ALWAYS check the conversation history first. 
2. If the user's latest request is a refinement (e.g., summarize, reformat, translate, change tone) of the information already provided in previous turns, DO NOT call any retrieval tools.
3. Use the tools ONLY when new information is needed that is not present in the history.
4. TOOL USAGE LIMIT: You should aim to answer in 1 or 2 tool calls. If a tool returns a definitive "not found", do not call it again.
5. If you have enough information to fulfill the user's request, STOP and provide the Final Answer immediately.

Example Answer:
# Overtime Policy Update 2025
FPT Software ensures fair compensation for all extra hours worked in accordance with local labor laws.

| Item | Detail | Summary |
|------|--------|---------|
| Monthly Limit | 60 hours max [1] | No one works >60h/mo without manager exception. |

### Internal Policy Analysis
According to the FSoft Human Rights Policy [1], employees must work overtime voluntarily. Pay is calculated at 150% of base rate for weekdays [2].
> **Source**: FSoft_Human Rights Policy

### External Market Context
Market trends show a shift toward "time-off-in-lieu" models [3], but FPT remains committed to the direct pay model to support employee earnings.

### Bottom Line
Log your hours in the portal by Friday and ensure your manager has signed off on the extra load.

### REFERENCE
1. FPT Software. (2024). FSoft Human Rights Policy (v2.1). Internal HR Document Repository.
2. National Assembly of Vietnam. (2019). Labor Code No. 45/2019/QH14. Government Portal. https://vanban.chinhphu.vn/
3. Society for Human Resource Management. (2025). 2025 Global Human Capital Trends Report. SHRM Insights. https://www.shrm.org/
"""

async def react_worker_node(state: AgentState) -> Dict[str, Any]:
    """
    Answer Synthesis node using the ReAct pattern.
    Supports streaming by yielding events to the main graph stream.
    """
    msg = state.get("msg", {})
    query = msg.get("msg_content", "")
    msg_id = msg.get("msg_id")
    history = state.get("messages", [])
    feedback = msg.get("msg_body", {}).get("human_feedback")
    
    llm = LLMService.get_chat_model()
    tools = [db_retrieval_tool, redis_retrieval_tool]
    
    # Use the prebuilt LangGraph ReAct agent which is more stable for internal loops
    agent = create_agent(llm, tools, system_prompt=REACT_SYSTEM_PROMPT.format(msg_id=msg_id))
    
    # Construct history-aware input
    if feedback:
        # If we have feedback, it's a revision. 
        # We append the feedback as a new instruction.
        input_messages = list(history) + [HumanMessage(content=f"Human Feedback/Refinement: {feedback}")]
    else:
        # Initial turn: history might have previous context if multi-turn session
        if history:
            input_messages = history + [HumanMessage(content=query)]
        else:
            input_messages = [HumanMessage(content=query)]
    
    # Process and return the state update with streaming
    final_messages = []
    from langchain_core.callbacks.manager import adispatch_custom_event
    
    first_chunk = True
    # We use multiple stream modes to capture intermediate token streams AND the final state
    async for chunk in agent.astream({"messages": input_messages}, stream_mode=["messages", "values"]):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
            if mode == "messages":
                msg_chunk, metadata = payload
                # Only stream content from the main agent node, exclude tool call metadata
                if metadata.get("langgraph_node") != "tools" and hasattr(msg_chunk, "content") and msg_chunk.content:
                    # Ignore tool execution blocks and just print the actual string content
                    has_tools = getattr(msg_chunk, "tool_calls", None) or getattr(msg_chunk, "tool_call_chunks", None)
                    if not has_tools:
                        text = msg_chunk.content
                        if first_chunk:
                            text = "\n" + text
                            first_chunk = False
                        await adispatch_custom_event("synthesis_stream", {"chunk": text})
            elif mode == "values":
                final_messages = payload.get("messages", [])
    
    if final_messages:
        final_answer = final_messages[-1].content
        msg.setdefault("msg_body", {})["answer"] = final_answer
        
        # Prepare harmonized data
        snippet = final_answer[:200] + ("..." if len(final_answer) > 200 else "")
        
        from src.core.schemas import AgenticPointer, ConversationArchive
        
        pointer = AgenticPointer(
            msg_id=msg_id,
            category="retrieval",
            snippet=snippet,
            status="final"
        )
        
        archive = ConversationArchive(
            msg_id=msg_id,
            category="retrieval",
            query=query,
            payload={"answer": final_answer},
            metadata={"timestamp": datetime.now().isoformat()}
        )

        # Update session memory index with the pointer
        indexing_update = {
            "retrieval_conv": {
                msg_id: pointer.model_dump()
            }
        }
        
        # Archive full content to Redis
        from src.services.redis_service import RedisService
        RedisService.cache_conversation(msg_id, archive.model_dump())
        
        # Sync pointer to Redis-backed agent memory index
        user_id = msg.get("user_id", "default_user")
        RedisService.update_index(user_id, "retrieval", pointer.model_dump())
        
        return {"msg": msg, "messages": final_messages, "summary": final_answer, **indexing_update}

    return {"msg": msg, "messages": final_messages}
