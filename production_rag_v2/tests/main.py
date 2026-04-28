import asyncio
import sys
import os
import logging
from pathlib import Path

# Ensure the 'production_rag_v2' root directory is in the path
sys.path.append(str(Path(__file__).parent.parent))

from langgraph.types import Command
from src.engine.graph_factory import create_rag_v2_graph
# from src.engine.state import create_initial_state
from src.services.queue_worker import QueueWorker
from src.services.memory_forge import MemoryForge, forge
from src.core.config import settings
settings.debug = True

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "interactive_debug.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

async def bootstrap_services():
    """Warms up all heavy ML services to ensure they are ready for the first request."""
    print("\n🚀 [Boost-up] Initializing production services...")
    
    # 1. Warm up Vector Service (FastEmbed)
    from src.services.vector_service import VectorService
    VectorService().warm_up()
    
    # 2. Warm up VnCoreNLP (Word Segmentation)
    from src.tools.vn_core_wrapper import VnCoreNLPWrapper
    VnCoreNLPWrapper().warm_up()
    
    # 3. Warm up LLM Service (Fireworks)
    from src.services.llm_service import LLMService
    LLMService.get_chat_model()

    # 4. Warm up Edge LLM (Receptionist)
    from src.services.edge_llm_service import EdgeLLMService
    EdgeLLMService.get_decision_model()
    
    print("✨ [Boost-up] Complete. System is now instant-ready.\n")

async def run_episodic_forger(forge: MemoryForge):
    """Starts the background episodic memory forger for agent"""
    await forge.start()

async def run_worker(worker: QueueWorker):
    """Starts the background search worker."""
    await worker.start()

async def shutdown(worker: QueueWorker, forger: MemoryForge, tasks: list):
    """Orchestrates a graceful shutdown of all services."""
    print("\n🛑 Shutting down services...")
    
    # 1. Stop background services
    worker.stop()
    forger.stop()
    
    # 2. Cancel all pending tasks
    for task in tasks:
        task.cancel()
    
    # Wait for tasks to finish cancellation
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # 3. Cleanup Services
    from src.services.edge_llm_service import EdgeLLMService
    from src.services.vector_service import VectorService
    from src.services.redis_service import redis_service
    from src.services.llm_service import LLMService

    EdgeLLMService.cleanup()
    await VectorService().cleanup()
    redis_service.cleanup()
    LLMService.cleanup()

    # 4. Final safety: Clean up any lingering aiohttp sessions (LangChain/LLM leaks)
    # Using a name-based check to avoid triggering 'FutureWarning' on complex objects (like torch)
    try:
        import aiohttp
        import gc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc.collect()
            for obj in gc.get_objects():
                try:
                    if obj.__class__.__name__ == 'ClientSession' and hasattr(obj, 'closed') and not obj.closed:
                        await obj.close()
                except Exception:
                    continue
    except Exception:
        pass

    print("✅ Shutdown complete.")

async def run_cli(worker: QueueWorker):
    """Main CLI loop for the RAG agent with proper async interrupt handling."""
    graph = create_rag_v2_graph()
    visible_nodes = {"receptionist", "react_worker", "approval", "scribing", "publishing", "meeting_node"}

    config = None  # Will be set per conversation turn
    
    print("\n🚀 FSoft Policy RAG Bot (Production v1) Started.")
    print("Type 'exit' or 'quit' to end the session.\n")

    conversation_count = 0
    session_id = "default_user"
    
    import os
    test_input_env = os.environ.get("TEST_INPUT", "")
    test_input_list = test_input_env.split("|") if test_input_env else []
    test_input_idx = 0

    try:
        while True: # while the program still run
            conversation_count += 1
            
            # Print a clear new-conversation banner every time we're ready for a new question
            print("─" * 60)
            print(f"  💬 New Conversation  #{conversation_count}")
            print("─" * 60)

            # Prevent blocking the event loop with synchronous input
            import os
            if 'test_input_list' not in locals():
                test_input_env = os.environ.get("TEST_INPUT", "")
                test_input_list = test_input_env.split("|") if test_input_env else []
                test_input_idx = 0
            
            if test_input_idx < len(test_input_list):
                user_input = test_input_list[test_input_idx].strip()
                test_input_idx += 1
                print(f"\nUser [AUTO]> {user_input}")
            else:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\nUser> ").strip()
                )

            # Session management logic for S-01 testing
            if user_input.startswith("/session"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    print("⚠️ Usage: /session <session_id>")
                    conversation_count -= 1
                    continue
                new_session = parts[1].strip()
                session_id = new_session
                print(f"👤 Switched to session: {session_id}")
                continue
            
            # Use the current session_id for the turn
            config = {"configurable": {"thread_id": session_id}}
            
            if user_input.lower() in ("exit", "quit", "stop", "bye"):
                print("👋 Goodbye!")
                return
                
            if not user_input:
                conversation_count -= 1  # Don't count empty inputs as a new conversation
                continue

            # Clean initialization for a fresh question
            current_input = {"msg": {"msg_content": user_input}}
            
            try:
                # Main Graph Stream Loop - handles one full question + HITL cycle
                while True:
                    print(f"📡 [Main] Starting graph stream with input: {current_input}")
                    async for event in graph.astream_events(
                        current_input, 
                        config=config, 
                        version="v2"
                    ):
                        # print(f"DEBUG: Event type: {event['event']}")
                        kind = event["event"]
                        
                        # 0. Handle Node Start (Headers)
                        if kind == "on_chain_start":
                            node_name = event.get("metadata", {}).get("langgraph_node", "")
                            if node_name == "react_worker":
                                print("\n🧠 [Agent Answer]:\n", flush=True)
                            elif node_name == "publishing":
                                print("\n📤 [Publishing Agent]:\n", flush=True)

                        # 1. Handle Custom Streaming (Receptionist & Synthesis)
                        elif kind == "on_custom_event" and event["name"] in ("receptionist_stream", "synthesis_stream", "publishing_stream"):
                            chunk = event["data"].get("chunk", "")
                            print(chunk, end="", flush=True)

                        # 2. Handle LLM Token Streaming (Expert Agent / Synthesis)
                        # 2. Handle Node Completion (top-level graph nodes only)
                        elif kind == "on_chain_end":
                            node_name = event.get("metadata", {}).get("langgraph_node", "unknown")
                            if node_name in visible_nodes:
                                if node_name in ("receptionist", "react_worker", "publishing"):
                                    print("\n", flush=True)
                                else:
                                    print(f"⚙️  [{node_name.title()}] Completed.")
                    
                    # Ensure we are on a new line after the stream finishes
                    print("\n")

                    # Check if we reached a human-in-the-loop checkpoint
                    state_snapshot = await graph.aget_state(config)
                    if state_snapshot.next: # does the graph reach the last-state == approval
                        # Extracts the last interrupt value for context
                        interrupts = state_snapshot.tasks[0].interrupts if state_snapshot.tasks else []
                        task_prompt = interrupts[0].value if interrupts else "Action required"
                        
                        if isinstance(task_prompt, dict):
                            q_text = task_prompt.get("question", "Feedback required:")
                            l_text = task_prompt.get("link")
                            if l_text:
                                q_text = f"{q_text}\n🔗 {l_text}"
                        else:
                            q_text = str(task_prompt)

                        print(f"✋ {q_text}")
                        
                        # Capture user approval or feedback
                        reply = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: input("> ").strip()
                        )
                        
                        # Resume graph with the user's response
                        current_input = Command(resume=reply)
                        continue  # Re-enter the astream loop with the resume command
                    
                    # No interrupt → graph reached END for this question cycle.
                    break
                    
            except Exception as e:
                print(f"❌ Error in Graph Turn: {e}")
    except asyncio.CancelledError:
        pass

async def main():
    # 🚀 Step 0: Boost-up heavy services
    await bootstrap_services()
    
    worker = QueueWorker()
    
    # Step 1: Create tasks
    cli_task = asyncio.create_task(run_cli(worker))
    worker_task = asyncio.create_task(run_worker(worker))
    forger_task = asyncio.create_task(run_episodic_forger(forge=forge))
    # Step 2: Wait for CLI to finish (either 'exit' command or interrupt)
    try:
        await cli_task
    finally:
        # Step 3: Trigger global shutdown
        await shutdown(worker, forge, [worker_task, forger_task])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass # Shutdown is handled in main()'s finally block
