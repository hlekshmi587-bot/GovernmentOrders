"""
Unit and integration tests for Phase 1 Step 1 components:
- AgentState
- OllamaClient
- LLMService
"""

import sys
from pydantic import BaseModel, Field

from app.agents.state import AgentState, ExecutionTraceItem
from app.agents.llm.ollama_client import OllamaClient
from app.agents.llm.llm_service import LLMService


class SampleIntentPlan(BaseModel):
    intent: str = Field(description="The identified user intent")
    target_go: str = Field(description="Target GO number mentioned")
    confidence: float = Field(description="Confidence value between 0.0 and 1.0")


def test_agent_state() -> None:
    print("\n--- Test 1: AgentState Initialization ---")
    state: AgentState = {
        "user_query": "Has G.O.(P) No.4/2026/P&ARD been amended?",
        "intent": "CHECK_AMENDMENT",
        "identified_go_numbers": ["G.O.(P) No.4/2026/P&ARD"],
        "discovered_go_numbers": [],
        "execution_plan": None,
        "selected_agents": ["search_agent", "relationship_agent"],
        "next_action": "search",
        "search_results": [],
        "document_analysis": [],
        "relationship_results": [],
        "verification_results": None,
        "evidence_sufficient": None,
        "confidence": None,
        "retry_count": 0,
        "max_retries": 2,
        "human_review_required": False,
        "human_review_status": "NONE",
        "human_review_notes": None,
        "human_feedback": None,
        "final_answer": None,
        "workflow_status": "INITIALIZED",
        "execution_trace": [
            {
                "timestamp": "2026-11-13T21:00:00",
                "agent": "supervisor",
                "action": "plan_initialized",
                "reasoning": "User asked to check amendment history",
                "tool_called": None,
                "tool_input": None,
                "tool_output_summary": None,
                "decision": "proceed_to_search",
            }
        ],
        "errors": [],
    }

    assert state["user_query"] is not None
    assert state["workflow_status"] == "INITIALIZED"
    assert len(state["execution_trace"]) == 1
    print("SUCCESS: AgentState initialized and validated successfully.")


def test_ollama_connectivity() -> OllamaClient:
    print("\n--- Test 2: Ollama Connectivity & Model Verification ---")
    client = OllamaClient()
    
    server_running = client.is_server_running()
    print(f"Ollama Server Running at {client.base_url}: {server_running}")
    assert server_running, "Ollama server is not running!"

    models = client.list_models()
    print(f"Available Models in Ollama: {models}")
    assert len(models) > 0, "No models found in Ollama!"

    has_llama = client.is_model_available("llama3.2:3b")
    print(f"Is llama3.2:3b available: {has_llama}")
    assert has_llama, "llama3.2:3b is not available in Ollama!"

    print("SUCCESS: Ollama server and llama3.2:3b model verified.")
    return client


def test_model_response(client: OllamaClient) -> None:
    print("\n--- Test 3: Raw Model Generation ---")
    prompt = "Answer in exactly 3 words: What is the capital of Kerala?"
    response = client.generate(prompt=prompt, temperature=0.0)
    print(f"Prompt: '{prompt}'")
    print(f"Model Response: '{response}'")
    assert len(response.strip()) > 0, "Model returned empty response!"
    print("SUCCESS: Model response generated successfully.")


def test_structured_output_parsing(client: OllamaClient) -> None:
    print("\n--- Test 4: Structured Output Parsing (JSON & Pydantic) ---")
    llm_service = LLMService(client=client, model="llama3.2:3b")

    prompt = (
        "Analyze this user query: 'Did G.O.(P) No.4/2026/P&ARD cancel any previous orders?'\n"
        "Extract:\n"
        "- intent: A short keyword like CHECK_CANCELLATION\n"
        "- target_go: The Government Order mentioned\n"
        "- confidence: A number between 0.0 and 1.0\n"
        "Output ONLY a valid JSON object matching these keys."
    )

    result = llm_service.generate_structured(
        prompt=prompt,
        schema_model=SampleIntentPlan,
        temperature=0.1,
    )

    print(f"Parsed Structured Output: {result}")
    assert isinstance(result, dict), "Result must be a dictionary!"
    assert "intent" in result, "Result missing 'intent' field!"
    assert "target_go" in result, "Result missing 'target_go' field!"
    assert "confidence" in result, "Result missing 'confidence' field!"
    print(f"Intent detected: {result['intent']}")
    print(f"Target GO: {result['target_go']}")
    print(f"Confidence: {result['confidence']}")
    print("SUCCESS: Structured JSON output parsed and validated with Pydantic.")


if __name__ == "__main__":
    try:
        test_agent_state()
        client = test_ollama_connectivity()
        test_model_response(client)
        test_structured_output_parsing(client)
        print("\n==========================================")
        print("ALL PHASE 1 STEP 1 TESTS PASSED SUCCESSFULLY!")
        print("==========================================")
    except Exception as exc:
        print(f"\nTEST FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
