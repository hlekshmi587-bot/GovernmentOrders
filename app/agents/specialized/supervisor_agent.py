"""
Supervisor Agent and Dynamic Planning Module.

This is the central decision-making and coordination component of the Multi-Agent System.
It uses the local LLM (llama3.2:3b) to understand natural language questions,
dynamically generate execution plans, select appropriate tools and agents,
and evaluate evidence to decide the next action or request human review.
"""

from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.agents.llm.llm_service import LLMService
from app.agents.state import AgentState

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Supported user query intents."""
    KEYWORD_SEARCH = "KEYWORD_SEARCH"
    GO_LOOKUP = "GO_LOOKUP"
    INCOMING_RELATIONSHIPS = "INCOMING_RELATIONSHIPS"
    OUTGOING_RELATIONSHIPS = "OUTGOING_RELATIONSHIPS"
    EXPLAIN_ORDER = "EXPLAIN_ORDER"
    AMBIGUOUS_CLARIFICATION = "AMBIGUOUS_CLARIFICATION"


ALLOWED_AGENTS = {
    "search_agent",
    "database_agent",
    "relationship_agent",
    "verification_agent",
    "response_agent",
    "human_review",
}

ALLOWED_TOOLS = {
    "search_tool",
    "database_tool",
    "relationship_tool",
    "none",
}

ALLOWED_NEXT_ACTIONS = {
    "search",
    "database",
    "relationship",
    "verification",
    "response",
    "human_review",
}

# Strict mapping of allowed parameters based on actual adapter method signatures:
# - SearchTool.search(keyword, limit)
# - DatabaseTool methods(go_number, direction, include_full_text, text_preview_chars)
# - RelationshipTool methods(go_number, text, source_go_number)
ALLOWED_TOOL_PARAMS: Dict[str, set[str]] = {
    "search_tool": {
        "keyword", "limit", "query", "retrieval_mode",
        "department_filter", "issue_date_start", "issue_date_end",
    },
    "database_tool": {
        "go_number", "query", "direction", "include_full_text",
        "text_preview_chars", "operation",
    },
    "relationship_tool": {
        "go_number", "query", "text", "source_go_number",
        "target_go_number", "mode", "min_confidence",
    },
    "none": set(),
}

DISALLOWED_PARAM_KEYWORDS = {
    "exec", "eval", "cmd", "sql", "command", "script", "code", "subprocess", "system", "os"
}


class PlannedStep(BaseModel):
    """A single execution step planned by the Supervisor."""
    step_number: int = Field(ge=1, description="Sequential step number starting at 1")
    action_name: str = Field(description="Short descriptive name of the action")
    target_agent: str = Field(description="Name of the agent responsible for this step")
    target_tool: str = Field(description="Tool to invoke: search_tool, database_tool, relationship_tool, or none")
    tool_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    expected_outcome: str = Field(description="What evidence this step is expected to collect")

    @field_validator("target_agent")
    @classmethod
    def validate_agent(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in ALLOWED_AGENTS:
            raise ValueError(f"Unknown agent '{v}'. Allowed agents: {sorted(ALLOWED_AGENTS)}")
        return clean_v

    @field_validator("target_tool")
    @classmethod
    def validate_tool(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in ALLOWED_TOOLS:
            raise ValueError(f"Unknown tool '{v}'. Allowed tools: {sorted(ALLOWED_TOOLS)}")
        return clean_v

    @field_validator("tool_parameters")
    @classmethod
    def validate_parameters(cls, params: Dict[str, Any], info) -> Dict[str, Any]:
        tool = info.data.get("target_tool")
        if not tool:
            return params

        tool_clean = tool.strip().lower()
        allowed_keys = ALLOWED_TOOL_PARAMS.get(tool_clean, set())

        for key, val in params.items():
            key_clean = key.strip().lower()
            if key_clean in DISALLOWED_PARAM_KEYWORDS:
                raise ValueError(f"Security violation: Disallowed parameter '{key}' in tool parameters.")
            if key_clean not in allowed_keys:
                raise ValueError(
                    f"Unsupported parameter '{key}' for tool '{tool}'. "
                    f"Allowed parameters: {sorted(allowed_keys)}"
                )
            if isinstance(val, str) and any(bad in val.lower() for bad in ["drop table", "select * from", "rm -rf", "exec("]):
                raise ValueError(f"Security violation: Suspicious command in parameter '{key}'.")

        # Specific parameter validations
        if tool_clean == "search_tool":
            if "retrieval_mode" in params:
                mode_v = str(params["retrieval_mode"]).strip().upper()
                if mode_v not in {"KEYWORD", "SEMANTIC", "HYBRID"}:
                    raise ValueError(f"Invalid retrieval_mode '{params['retrieval_mode']}'. Allowed: KEYWORD, SEMANTIC, HYBRID")
            if "limit" in params:
                lim = params["limit"]
                if isinstance(lim, bool) or not isinstance(lim, int) or lim < 1 or lim > 50:
                    raise ValueError(f"limit must be an integer between 1 and 50, got {lim}")
            if "query" in params:
                q = params["query"]
                if not isinstance(q, str) or not q.strip():
                    raise ValueError("query cannot be empty or whitespace.")
            if "keyword" in params:
                kw = params["keyword"]
                if not isinstance(kw, str) or not kw.strip():
                    raise ValueError("keyword cannot be empty or whitespace.")
            if "department_filter" in params and params["department_filter"] is not None:
                dept = params["department_filter"]
                if not isinstance(dept, str) or not dept.strip():
                    raise ValueError("department_filter must be a non-empty string or None.")
            if "issue_date_start" in params and params["issue_date_start"] is not None:
                ds = str(params["issue_date_start"])
                try:
                    datetime.strptime(ds, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"issue_date_start must be YYYY-MM-DD, got {ds}")
            if "issue_date_end" in params and params["issue_date_end"] is not None:
                de = str(params["issue_date_end"])
                try:
                    datetime.strptime(de, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"issue_date_end must be YYYY-MM-DD, got {de}")
            if params.get("issue_date_start") and params.get("issue_date_end"):
                ds = str(params["issue_date_start"])
                de = str(params["issue_date_end"])
                if ds > de:
                    raise ValueError(f"issue_date_start ({ds}) cannot be later than issue_date_end ({de}).")

        elif tool_clean == "database_tool":
            if "operation" in params:
                op_v = str(params["operation"]).strip().upper()
                if op_v not in {"LOOKUP_ORDER", "GET_METADATA", "GET_REFERENCES", "GET_RELATIONSHIPS", "LIST_ALL_ORDERS"}:
                    raise ValueError(f"Invalid database operation '{params['operation']}'.")
            if "direction" in params:
                dir_v = str(params["direction"]).strip().upper()
                if dir_v not in {"INCOMING", "OUTGOING", "ALL"}:
                    raise ValueError(f"Invalid relationship direction '{params['direction']}'.")
            if "include_full_text" in params:
                ift = params["include_full_text"]
                if not isinstance(ift, bool):
                    raise ValueError(f"include_full_text must be a strict boolean, got {type(ift).__name__}")
            if "text_preview_chars" in params:
                tpc = params["text_preview_chars"]
                if isinstance(tpc, bool) or not isinstance(tpc, int) or tpc < 0 or tpc > 50000:
                    raise ValueError(f"text_preview_chars must be an integer between 0 and 50000, got {tpc}")
            if "go_number" in params and params["go_number"] is not None:
                gon = params["go_number"]
                if not isinstance(gon, str) or not gon.strip():
                    raise ValueError("go_number cannot be empty or whitespace.")

        elif tool_clean == "relationship_tool":
            if "mode" in params:
                mode_v = str(params["mode"]).strip().upper()
                if mode_v not in {"INVESTIGATE_ORDER", "EXTRACT_FROM_TEXT", "INVESTIGATE_PAIR", "CROSS_EXAMINE_DATABASE"}:
                    raise ValueError(f"Invalid relationship mode '{params['mode']}'.")
            if "min_confidence" in params and params["min_confidence"] is not None:
                mc_v = str(params["min_confidence"]).strip().upper()
                if mc_v not in {"HIGH", "MEDIUM", "LOW"}:
                    raise ValueError(f"Invalid min_confidence '{params['min_confidence']}'.")
            # Strict conflicting parameter check
            if "source_go_number" in params and "go_number" in params:
                s_go = str(params["source_go_number"]).strip().lower()
                g_go = str(params["go_number"]).strip().lower()
                if s_go != g_go:
                    raise ValueError(
                        f"Conflicting parameters: 'source_go_number' ('{params['source_go_number']}') "
                        f"and 'go_number' ('{params['go_number']}') have different values."
                    )
            if "go_number" in params and params["go_number"] is not None:
                gon = params["go_number"]
                if not isinstance(gon, str) or not gon.strip():
                    raise ValueError("go_number cannot be empty or whitespace.")
            if "source_go_number" in params and params["source_go_number"] is not None:
                sgon = params["source_go_number"]
                if not isinstance(sgon, str) or not sgon.strip():
                    raise ValueError("source_go_number cannot be empty or whitespace.")
            if "target_go_number" in params and params["target_go_number"] is not None:
                tgon = params["target_go_number"]
                if not isinstance(tgon, str) or not tgon.strip():
                    raise ValueError("target_go_number cannot be empty or whitespace.")
            if "text" in params and params["text"] is not None:
                txt = params["text"]
                if not isinstance(txt, str) or not txt.strip():
                    raise ValueError("text cannot be empty or whitespace.")

        return params


class SupervisorPlan(BaseModel):
    """
    Structured dynamic execution plan generated by the Supervisor.
    """
    intent: QueryIntent
    identified_go_numbers: List[str] = Field(default_factory=list, description="Target GO numbers extracted from query")
    identified_departments: List[str] = Field(default_factory=list, description="Target departments extracted from query")
    reasoning: str = Field(description="Explanation of why this approach and tools were selected")
    selected_agents: List[str] = Field(description="List of specialized agents needed")
    selected_tools: List[str] = Field(description="List of tools needed")
    execution_steps: List[PlannedStep] = Field(description="Ordered list of execution steps")
    expected_evidence: str = Field(description="Description of what facts are required to fully answer the query")
    next_action: str = Field(description="Immediate next routing action: search, database, relationship, verification, response, human_review")
    confidence: float = Field(ge=0.0, le=1.0, description="Planning confidence score between 0.0 and 1.0")
    human_review_required: bool = Field(default=False, description="Flag indicating human review is needed immediately")
    plan_source: str = Field(default="LLM", description="Source of the plan: 'LLM' or 'HEURISTIC_FALLBACK'")

    @field_validator("next_action")
    @classmethod
    def validate_next_action(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in ALLOWED_NEXT_ACTIONS:
            raise ValueError(f"Unknown next_action '{v}'. Allowed: {sorted(ALLOWED_NEXT_ACTIONS)}")
        return clean_v

    @field_validator("selected_agents")
    @classmethod
    def validate_agents(cls, v: List[str]) -> List[str]:
        cleaned = [item.strip().lower() for item in v]
        for a in cleaned:
            if a not in ALLOWED_AGENTS:
                raise ValueError(f"Unknown agent '{a}' in selected_agents.")
        return cleaned

    @field_validator("selected_tools")
    @classmethod
    def validate_tools(cls, v: List[str]) -> List[str]:
        cleaned = [item.strip().lower() for item in v]
        for t in cleaned:
            if t not in ALLOWED_TOOLS:
                raise ValueError(f"Unknown tool '{t}' in selected_tools.")
        return cleaned

    @field_validator("plan_source")
    @classmethod
    def validate_plan_source(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in ("LLM", "HEURISTIC_FALLBACK"):
            raise ValueError(f"Invalid plan_source '{v}'. Must be 'LLM' or 'HEURISTIC_FALLBACK'.")
        return clean_v

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> "SupervisorPlan":
        """Validate step sequentiality, agent/tool pairings, and coherence between steps and overall plan."""
        steps = self.execution_steps

        # Never override a valid human-review decision
        if self.human_review_required or self.intent == QueryIntent.AMBIGUOUS_CLARIFICATION:
            self.human_review_required = True
            self.next_action = "human_review"
            return self

        if not steps:
            return self

        # 1. Validate sequential step numbers: 1, 2, 3...
        expected_numbers = list(range(1, len(steps) + 1))
        actual_numbers = [s.step_number for s in steps]
        if actual_numbers != expected_numbers:
            raise ValueError(f"Step numbers must be strictly sequential starting at 1. Found {actual_numbers}, expected {expected_numbers}.")

        # 2. Strict valid agent/tool pairings mapping to authorized routing actions
        valid_agent_tool_map: Dict[tuple[str, str], str] = {
            ("search_agent", "search_tool"): "search",
            ("database_agent", "database_tool"): "database",
            ("relationship_agent", "relationship_tool"): "relationship",
            ("human_review", "none"): "human_review",
            ("verification_agent", "none"): "verification",
            ("response_agent", "none"): "response",
        }

        # Validate every step's agent/tool pairing - reject unsupported pairings
        for s in steps:
            agent_clean = s.target_agent.strip().lower()
            tool_clean = s.target_tool.strip().lower()
            if (agent_clean, tool_clean) not in valid_agent_tool_map:
                raise ValueError(
                    f"Invalid agent/tool pairing in Step {s.step_number}: agent '{agent_clean}' cannot use tool '{tool_clean}'. "
                    f"Authorized pairings: {sorted(list(valid_agent_tool_map.keys()))}"
                )

        # 3. Ensure every step's agent and tool are registered in selected_agents and selected_tools (auto-heal)
        agents_set = set(self.selected_agents)
        tools_set = set(self.selected_tools)
        for s in steps:
            if s.target_agent not in agents_set:
                self.selected_agents.append(s.target_agent)
                agents_set.add(s.target_agent)
            if s.target_tool not in tools_set:
                self.selected_tools.append(s.target_tool)
                tools_set.add(s.target_tool)

        # 4. Auto-normalize INCOMING_RELATIONSHIPS parameters for database_agent
        if self.intent == QueryIntent.INCOMING_RELATIONSHIPS:
            for s in steps:
                if s.target_agent.strip().lower() == "database_agent" and s.target_tool.strip().lower() == "database_tool":
                    if not s.tool_parameters.get("operation"):
                        s.tool_parameters["operation"] = "GET_RELATIONSHIPS"
                    if not s.tool_parameters.get("direction"):
                        s.tool_parameters["direction"] = "INCOMING"

        # 5. Derive expected routing action from the first execution step only when its pairing is valid
        first_step = steps[0]
        first_pair = (first_step.target_agent.strip().lower(), first_step.target_tool.strip().lower())
        derived_action = valid_agent_tool_map[first_pair]

        # If next_action conflicts with that valid pairing, align it with the derived action
        if self.next_action != derived_action:
            self.next_action = derived_action

        return self


class SupervisorReplanDecision(BaseModel):
    """
    Structured decision returned when the Supervisor evaluates collected evidence.
    """
    evidence_sufficient: bool = Field(description="True if collected evidence is sufficient to answer the user query")
    reasoning: str = Field(description="Reasoning explaining the evidence evaluation or replanning decision")
    next_action: str = Field(description="Next routing destination: verification, response, search, database, relationship, or human_review")
    suggested_tool: Optional[str] = Field(default=None, description="Tool to invoke if replanning/retrying")
    suggested_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Parameters for the retry tool")
    human_review_required: bool = Field(default=False, description="Set to True if ambiguity cannot be resolved automatically")
    human_review_reason: Optional[str] = Field(default=None, description="Note explaining why human review is required")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in current findings (0.0 to 1.0)")
    decision_source: str = Field(default="LLM", description="Source of this replanning decision: 'LLM' or 'HEURISTIC_FALLBACK'")

    @field_validator("next_action")
    @classmethod
    def validate_next_action(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in ALLOWED_NEXT_ACTIONS:
            raise ValueError(f"Unknown next_action '{v}'. Allowed: {sorted(ALLOWED_NEXT_ACTIONS)}")
        return clean_v

    @field_validator("decision_source")
    @classmethod
    def validate_decision_source(cls, v: str) -> str:
        clean_v = v.strip().upper()
        if clean_v not in ("LLM", "HEURISTIC_FALLBACK"):
            raise ValueError(f"Invalid decision_source '{v}'. Must be 'LLM' or 'HEURISTIC_FALLBACK'.")
        return clean_v


class SupervisorAgent:
    """
    Supervisor Agent coordinating specialized agents and tools using LangGraph.
    """

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        """
        Initialize the Supervisor Agent.

        Args:
            llm_service: LLMService instance for local LLM reasoning.
        """
        self.llm = llm_service or LLMService()

    def create_initial_plan(self, user_query: str) -> SupervisorPlan:
        """
        Analyze a user query and generate a typed, dynamic execution plan.
        Uses unbiased prompting so the LLM independently selects the correct intent and tools.

        Args:
            user_query: The natural-language query from the user.

        Returns:
            Validated SupervisorPlan instance.
        """
        cleaned_query = user_query.strip() if user_query else ""
        if not cleaned_query:
            return SupervisorPlan(
                intent=QueryIntent.AMBIGUOUS_CLARIFICATION,
                identified_go_numbers=[],
                identified_departments=[],
                reasoning="Empty query provided; human clarification required.",
                selected_agents=["human_review"],
                selected_tools=["none"],
                execution_steps=[],
                expected_evidence="A valid user question",
                next_action="human_review",
                confidence=0.0,
                human_review_required=True,
                plan_source="HEURISTIC_FALLBACK",
            )

        # Unbiased, streamlined prompt: explicitly defines all 6 intents, allowed components, and routing constraints
        prompt = f"""You are the Supervisor Agent for an AI system analyzing Kerala Government Orders.
Analyze the user's natural language question and create a structured execution plan.

USER QUERY:
"{cleaned_query}"

PLANNING GUIDELINES:
1. Intent Definitions & Routing Rules:
   - KEYWORD_SEARCH: General search by topic, subject, department, policy, pension, or allowance without an explicit GO identifier (e.g., "Find orders regarding dearness relief and pension revision", "Finance Department", "Revenue Department", "Find orders from P&ARD"). Words like "revision", "modified", "order", "allowance", or "relief" in a topic query do NOT make it a relationship query unless a specific GO identifier is investigated.
     -> target_agent: "search_agent", target_tool: "search_tool" (params: {{"keyword": "..."}}), next_action: "search"
   - GO_LOOKUP: Direct request to show, view, or retrieve a specified Government Order without relationship questions (e.g., "Show details for <GO_IDENTIFIER>", "View order <GO_IDENTIFIER>", or simply "<GO_IDENTIFIER>"). Do NOT use if the user asks which orders modified or are modified by this order.
     -> target_agent: "database_agent", target_tool: "database_tool" (params: {{"go_number": "<GO_IDENTIFIER>"}}), next_action: "database"
   - INCOMING_RELATIONSHIPS: User asks which LATER order modified, amended, cancelled, or superseded a specified order, or asks for incoming relationships pointing to a specified order (e.g., "Show incoming relationships for <GO_IDENTIFIER>", "Which later order modified <GO_IDENTIFIER>?"). Requires a specific, non-empty GO identifier.
     Single-step plan:
     * Step 1: database_agent / database_tool (params: {{"go_number": "<GO_IDENTIFIER>", "operation": "GET_RELATIONSHIPS", "direction": "INCOMING"}})
     -> next_action: "database"
   - OUTGOING_RELATIONSHIPS: User asks what earlier orders a specified order modifies, amends, cancels, or references (e.g., "Which orders were modified by <GO_IDENTIFIER>?", "What earlier orders does <GO_IDENTIFIER> amend?"). Requires a specific, non-empty GO identifier.
     -> target_agent: "relationship_agent", target_tool: "relationship_tool" (params: {{"go_number": "<GO_IDENTIFIER>"}}), next_action: "relationship"
   - EXPLAIN_ORDER: Detailed clause-by-clause analysis of an order and its changes. Requires a specific GO identifier.
     Multi-step plan required:
     * Step 1: database_agent / database_tool (params: {{"go_number": "<GO_IDENTIFIER>", "include_full_text": true}})
     * Step 2: relationship_agent / relationship_tool (params: {{"go_number": "<GO_IDENTIFIER>"}})
     -> next_action: "database" (matches Step 1)
   - AMBIGUOUS_CLARIFICATION: Query is too vague, ambiguous, or incomplete to act on safely (e.g., "check the order", "what happened?").
     -> target_agent: "human_review", target_tool: "none", next_action: "human_review", human_review_required: true

2. Core Constraints & Allowed Components:
   - Relationship verbs ("modified by", "which later order modified", "amended by") determine relationship intent; presence of a GO identifier alone does NOT determine intent.
   - For queries without a specific GO identifier: select KEYWORD_SEARCH if the query has a clear topic/subject/department, or AMBIGUOUS_CLARIFICATION if it is too vague to act on safely. Never leave go_number empty "" in database_tool or relationship_tool.
   - Allowed agents: ["search_agent", "database_agent", "relationship_agent", "verification_agent", "response_agent", "human_review"].
   - Allowed tools: ["search_tool", "database_tool", "relationship_tool", "none"].
   - Allowed next_action values: ["search", "database", "relationship", "verification", "response", "human_review"].
   - Strict agent/tool pairings:
     * database_agent MUST use database_tool
     * search_agent MUST use search_tool
     * relationship_agent MUST use relationship_tool
     * human_review MUST use tool "none"
   - Permitted tool parameters:
     * search_tool: "keyword", "limit"
     * database_tool: "go_number", "operation", "direction", "include_full_text"
     * relationship_tool: "go_number", "text"
   - Step numbers in execution_steps must be strictly sequential (1, 2, ...).
   - "next_action" must strictly match the tool of step 1: database_tool -> "database", search_tool -> "search", relationship_tool -> "relationship", none -> "human_review".

OUTPUT FORMAT:
Respond with a strictly valid JSON object matching this schema:
{{
  "intent": "<ONE_OF_THE_6_INTENTS_ABOVE>",
  "identified_go_numbers": ["<EXTRACTED_GO_NUMBERS_OR_EMPTY_LIST>"],
  "identified_departments": ["<EXTRACTED_DEPARTMENTS_OR_EMPTY_LIST>"],
  "reasoning": "<EXPLANATION_OF_WHY_THIS_PLAN_WAS_CHOSEN>",
  "selected_agents": ["<LIST_OF_AGENTS_USED_IN_STEPS>"],
  "selected_tools": ["<LIST_OF_TOOLS_USED_IN_STEPS>"],
  "execution_steps": [
    {{
      "step_number": 1,
      "action_name": "<ACTION_NAME>",
      "target_agent": "<AGENT_NAME>",
      "target_tool": "<TOOL_NAME>",
      "tool_parameters": {{"param_key": "param_value"}},
      "expected_outcome": "<EXPECTED_OUTCOME>"
    }}
  ],
  "expected_evidence": "<DESCRIPTION_OF_REQUIRED_EVIDENCE>",
  "next_action": "<NEXT_ACTION_MATCHING_STEP_1>",
  "confidence": 0.9,
  "human_review_required": false,
  "plan_source": "LLM"
}}

Respond strictly with valid JSON conforming to this structure."""

        system_prompt = (
            "You are a Senior Agentic AI Architect and Coordinator. "
            "You dynamically plan multi-agent workflows for legal and government order analysis. "
            "Always return clean, structured JSON conforming to the requested schema."
        )

        try:
            plan_dict = self.llm.generate_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                schema_model=SupervisorPlan,
                temperature=0.1,
                max_retries=1,
                num_predict=350,
                num_ctx=2048,
            )
            plan_dict["plan_source"] = "LLM"
            return SupervisorPlan.model_validate(plan_dict)

        except Exception as exc:
            logger.warning("LLM planning fallback triggered due to error: %s", exc)
            return self._heuristic_fallback_plan(cleaned_query)

    def _heuristic_fallback_plan(self, query: str) -> SupervisorPlan:
        """
        Deterministic safeguard fallback if local LLM fails to output valid JSON.
        Explicitly marked with plan_source='HEURISTIC_FALLBACK'.
        """
        import re
        lower = query.lower()

        # Extract GO identifier if present in query sentence (supports full formats and shorthand like G.O. 114)
        go_match = re.search(
            r"G\.?\s*O\.?\s*(?:\([A-Za-z]+\))?\s*(?:No\.?)?\s*\d+(?:/\d{4}/[A-Za-z&.-]+)?|"
            r"സ\s*\.?\s*ഉ\s*\.?\s*(?:\([^)]*\))?\s*(?:നം|നമ്പർ)?\s*\.?\s*\d+(?:/\d{4}/[\w&.-]+)?|"
            r"\b\d+/\d{4}/[A-Za-z&.-]+",
            query,
            re.IGNORECASE,
        )
        extracted_go = [go_match.group(0).rstrip(".?, ")] if go_match else []
        identified_depts: List[str] = []

        # Heuristic intent classification:
        # Require a successfully extracted GO identifier before selecting intents that require a specific GO number
        if extracted_go:
            go_num = extracted_go[0]
            if any(k in lower for k in [
                "incoming",
                "incoming relationship",
                "incoming relationships",
                "orders that modify",
                "orders that reference",
                "which later",
                "later order",
                "who modified",
                "subsequent order",
                "later government order",
            ]):
                intent = QueryIntent.INCOMING_RELATIONSHIPS
                action = "database"
                tool = "database_tool"
                agent = "database_agent"
                params = {"go_number": go_num, "operation": "GET_RELATIONSHIPS", "direction": "INCOMING"}
                steps = [
                    PlannedStep(
                        step_number=1,
                        action_name="lookup_incoming_relationships",
                        target_agent=agent,
                        target_tool=tool,
                        tool_parameters=params,
                        expected_outcome="Records of later orders modifying this order",
                    )
                ]
                agents = ["database_agent", "verification_agent", "response_agent"]
                tools = ["database_tool"]
            elif any(k in lower for k in ["explain", "clause", "changes introduced by"]):
                intent = QueryIntent.EXPLAIN_ORDER
                action = "database"
                steps = [
                    PlannedStep(
                        step_number=1,
                        action_name="retrieve_full_text",
                        target_agent="database_agent",
                        target_tool="database_tool",
                        tool_parameters={"go_number": go_num, "include_full_text": True},
                        expected_outcome="Complete document text",
                    ),
                    PlannedStep(
                        step_number=2,
                        action_name="extract_clauses_and_references",
                        target_agent="relationship_agent",
                        target_tool="relationship_tool",
                        tool_parameters={"go_number": go_num},
                        expected_outcome="Extracted clauses and references",
                    ),
                ]
                agents = ["database_agent", "relationship_agent", "verification_agent", "response_agent"]
                tools = ["database_tool", "relationship_tool"]
            elif any(k in lower for k in ["earlier", "outgoing", "amends", "modifies", "modified by", "amended by", "cancelled by", "superseded by", "references"]):
                intent = QueryIntent.OUTGOING_RELATIONSHIPS
                action = "database"
                steps = [
                    PlannedStep(
                        step_number=1,
                        action_name="lookup_outgoing_relationships",
                        target_agent="database_agent",
                        target_tool="database_tool",
                        tool_parameters={"go_number": go_num, "direction": "OUTGOING"},
                        expected_outcome="Records of earlier orders modified by this order",
                    )
                ]
                agents = ["database_agent", "verification_agent", "response_agent"]
                tools = ["database_tool"]
            else:
                intent = QueryIntent.GO_LOOKUP
                action = "database"
                steps = [
                    PlannedStep(
                        step_number=1,
                        action_name="lookup_order_by_number",
                        target_agent="database_agent",
                        target_tool="database_tool",
                        tool_parameters={"go_number": go_num},
                        expected_outcome="Specific Government Order metadata and text",
                    )
                ]
                agents = ["database_agent", "response_agent"]
                tools = ["database_tool"]
        elif (
            not lower.strip()
            or re.sub(r"[^\w\s]", "", lower).strip() in {
                "hi", "hello", "hey", "help", "thanks", "thank you",
                "thankyou", "thanks a lot", "please help", "help me",
                "check order", "check the order", "what happened",
                "test",
            }
            or (
                len(re.sub(r"[^\w\s]", "", lower).strip().split()) <= 2
                and all(
                    t in {"hi", "hello", "hey", "help", "thanks", "thank", "you", "please"}
                    for t in re.sub(r"[^\w\s]", "", lower).strip().split()
                )
            )
        ):
            intent = QueryIntent.AMBIGUOUS_CLARIFICATION
            action = "human_review"
            steps = [
                PlannedStep(
                    step_number=1,
                    action_name="request_clarification",
                    target_agent="human_review",
                    target_tool="none",
                    tool_parameters={},
                    expected_outcome="Clarified user question",
                )
            ]
            agents = ["human_review"]
            tools = ["none"]
            identified_depts = []
        else:
            intent = QueryIntent.KEYWORD_SEARCH
            action = "search"
            identified_depts = []
            for d_kw, d_name in [
                ("finance", "Finance Department"),
                ("revenue", "Revenue Department"),
                ("health", "Health Department"),
                ("education", "General Education Department"),
                ("transport", "Transport Department"),
                ("p&ard", "Personnel and Administrative Reforms Department"),
                ("itd", "Electronics and Information Technology Department"),
                ("lsgd", "Local Self Government Department"),
                ("taxes", "Taxes Department"),
                ("law", "Law Department"),
            ]:
                if d_kw in lower:
                    identified_depts.append(d_name)

            steps = [
                PlannedStep(
                    step_number=1,
                    action_name="keyword_search_orders",
                    target_agent="search_agent",
                    target_tool="search_tool",
                    tool_parameters={"keyword": query},
                    expected_outcome="Matching Government Orders from PostgreSQL",
                )
            ]
            agents = ["search_agent", "response_agent"]
            tools = ["search_tool"]

        return SupervisorPlan(
            intent=intent,
            identified_go_numbers=extracted_go,
            identified_departments=identified_depts,
            reasoning=f"Heuristic fallback plan generated for query: {query}",
            selected_agents=agents,
            selected_tools=tools,
            execution_steps=steps,
            expected_evidence="Verified database records matching the query",
            next_action=action,
            confidence=0.7,
            human_review_required=(action == "human_review"),
            plan_source="HEURISTIC_FALLBACK",
        )

    def evaluate_and_replan(self, state: AgentState) -> SupervisorReplanDecision:
        """
        Examine collected evidence in state and decide whether to continue,
        retry with another tool, request human review, or proceed to response.

        Distinguishes raw collected candidates from verified facts.

        Args:
            state: Current AgentState.

        Returns:
            Validated SupervisorReplanDecision instance.
        """
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        search_results = state.get("search_results", [])
        document_analysis = state.get("document_analysis", [])
        relationship_results = state.get("relationship_results", [])
        errors = state.get("errors", [])
        user_query = state.get("user_query", "")
        verification_status = state.get("verification_status")
        verification_results = state.get("verification_results")
        verification_already_run = verification_status is not None

        # 1. Deterministic Safeguard: Check for Ambiguity
        for res in document_analysis + relationship_results + search_results:
            if isinstance(res, dict) and res.get("status") in ("AMBIGUOUS_MATCH", "AMBIGUOUS_SOURCE_ORDER"):
                candidates = res.get("candidates", [])
                return SupervisorReplanDecision(
                    evidence_sufficient=False,
                    reasoning=f"Multiple Government Orders match the query: {candidates}. Human review is required to resolve ambiguity.",
                    next_action="human_review",
                    human_review_required=True,
                    human_review_reason=f"Ambiguous order matches: {candidates}",
                    confidence=0.3,
                    decision_source="HEURISTIC_FALLBACK",
                )

        # 2. Check for Unverified Relationships: Raw pattern matches or database candidates are NOT legally verified facts
        # If relationship evidence exists and verification hasn't run yet, route to verification node
        has_unverified = False
        if relationship_results and not verification_already_run:
            plan = state.get("execution_plan") or {}
            raw_intent = state.get("intent") or plan.get("intent")
            intent_name = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent or "")
            is_rel_intent = intent_name in (
                QueryIntent.INCOMING_RELATIONSHIPS.value,
                QueryIntent.OUTGOING_RELATIONSHIPS.value,
                QueryIntent.EXPLAIN_ORDER.value,
                "INCOMING_RELATIONSHIPS",
                "OUTGOING_RELATIONSHIPS",
                "EXPLAIN_ORDER",
            )
            for rel in relationship_results:
                if isinstance(rel, dict):
                    v_status = rel.get("verification_status")
                    legal_verified = rel.get("legal_effect_verified", False)
                    # Catches both explicit UNVERIFIED_CANDIDATE and database candidates where legal_effect_verified is False or verification_status is unset
                    if v_status == "UNVERIFIED_CANDIDATE" or not legal_verified or (is_rel_intent and v_status is None):
                        has_unverified = True
                        break

        if has_unverified and not verification_already_run:
            return SupervisorReplanDecision(
                evidence_sufficient=False,
                reasoning="Extracted relationships are unverified candidates (UNVERIFIED_CANDIDATE). Routing to verification node to check evidence against database records.",
                next_action="verification",
                confidence=0.6,
                decision_source="HEURISTIC_FALLBACK",
            )

        # 3. Deterministic Safeguard: Verification Cycle Prevention
        # If verification has already executed on the collected evidence, prevent endless cycling
        if verification_already_run:
            is_suff = bool(state.get("evidence_sufficient", False))
            return SupervisorReplanDecision(
                evidence_sufficient=is_suff,
                reasoning=f"Evidence verification has completed (status: {verification_status}). Concluding investigation and synthesizing response.",
                next_action="response",
                confidence=0.85 if is_suff else 0.5,
                decision_source="HEURISTIC_FALLBACK",
            )

        # 4. Deterministic Safeguard: Bounded Retries Exhaustion
        if retry_count >= max_retries:
            logger.info("Maximum retries (%d) reached.", max_retries)
            has_some_data = bool(search_results or document_analysis or relationship_results)
            return SupervisorReplanDecision(
                evidence_sufficient=has_some_data,
                reasoning=f"Reached maximum retry limit ({max_retries}). Concluding investigation with available evidence.",
                next_action="response" if (has_some_data and verification_already_run) else ("verification" if has_some_data else "human_review"),
                human_review_required=not has_some_data,
                human_review_reason="No sufficient evidence gathered within maximum retry attempts." if not has_some_data else None,
                confidence=0.5 if has_some_data else 0.2,
                decision_source="HEURISTIC_FALLBACK",
            )

        # 5. Check for Empty or Failed Results: Do NOT repeat the exact same failed action!
        # Check if database lookup returned NOT_FOUND
        db_failed = False
        for doc in document_analysis:
            if isinstance(doc, dict) and doc.get("status") == "NOT_FOUND":
                db_failed = True
                break

        if db_failed:
            # Instead of repeating database_tool on the exact same missing GO, replan to broader search_tool
            return SupervisorReplanDecision(
                evidence_sufficient=False,
                reasoning="Direct database lookup returned NOT_FOUND. Replanning to search_tool with keyword search rather than repeating failed lookup.",
                next_action="search",
                suggested_tool="search_tool",
                suggested_parameters={"keyword": user_query},
                confidence=0.5,
                decision_source="HEURISTIC_FALLBACK",
            )

        # Check if search returned NO_MATCH or 0 records
        search_empty = False
        for s in search_results:
            if isinstance(s, dict) and (s.get("status") == "NO_MATCH" or s.get("returned_count") == 0):
                search_empty = True
                break

        if search_empty and not (document_analysis or relationship_results):
            return SupervisorReplanDecision(
                evidence_sufficient=False,
                reasoning="Keyword search returned no matching records. Routing to human review for query clarification.",
                next_action="human_review",
                human_review_required=True,
                human_review_reason=f"No matching records found for query: '{user_query}'",
                confidence=0.2,
                decision_source="HEURISTIC_FALLBACK",
            )

        # 6. Check if we have verified evidence
        has_primary_evidence = bool(search_results or document_analysis or relationship_results)

        if not has_primary_evidence and not errors:
            return SupervisorReplanDecision(
                evidence_sufficient=False,
                reasoning="Initial tool execution has not returned results yet. Proceeding with planned action.",
                next_action=state.get("next_action") or "search",
                confidence=0.6,
                decision_source="HEURISTIC_FALLBACK",
            )

        # 7. LLM Evidence Sufficiency Evaluation
        plan = state.get("execution_plan") or {}
        expected_evidence = plan.get("expected_evidence", "Sufficient information to answer user query")
        current_intent = state.get("intent") or plan.get("intent")
        intent_str = current_intent.value if hasattr(current_intent, "value") else str(current_intent or "GENERAL_INQUIRY")

        summary_data = {
            "search_matches": len(search_results),
            "documents_analyzed": len(document_analysis),
            "relationships_found": len(relationship_results),
            "verification_performed": verification_already_run,
            "verification_status": verification_status,
            "sample_evidence": [
                r.get("evidence_text")
                for r in relationship_results
                if isinstance(r, dict) and r.get("evidence_text")
            ][:3],
        }

        prompt = f"""You are evaluating evidence for a Kerala Government Order inquiry.
USER QUERY: "{user_query}"
CURRENT INTENT: "{intent_str}"
EXPECTED EVIDENCE: "{expected_evidence}"
COLLECTED SUMMARY: {summary_data}

EVALUATION GUIDELINES:
1. Permitted next_action values: "verification", "response", "database", "search", "relationship", "human_review".
2. Verification and Response Routing Rules:
   - When evidence is sufficient to answer the question, route to "verification" if verification has NOT yet run (verification_performed is false).
   - Direct routing to "response" is permitted ONLY after verification has completed (verification_performed is true).
   - When proceeding to "verification" or "response", set suggested_tool: null, suggested_parameters: null, and confidence: >= 0.8.
3. Query-specific guidance:
   - For relationship intents (INCOMING_RELATIONSHIPS, OUTGOING_RELATIONSHIPS, EXPLAIN_ORDER): Retrieved relationship evidence from database or text analysis MUST be sent to Verification Agent (next_action="verification") before response synthesis. Direct response without verification is strictly forbidden.
   - For KEYWORD_SEARCH: When relevant search matches exist in search_matches and no relationship or document analysis claims are present, document discovery is complete. Route directly to "response" (next_action="response", evidence_sufficient=true). Do not route to verification because ordinary keyword searches do not make specific inter-order claims.
   - For GO_LOOKUP: Preserve existing database lookup and verification workflow.
   - CRITICAL: Never select "verification" if verification has already run (verification_performed is true) without a new retrieval step.
4. If evidence is incomplete and a specific retry tool can gather missing facts:
   - evidence_sufficient: false
   - next_action: "database", "search", or "relationship"
   - suggested_tool: "database_tool", "search_tool", or "relationship_tool"
   - suggested_parameters: parameters dict for the suggested tool
   - confidence: 0.4 - 0.7
5. If findings are ambiguous or blocked:
   - evidence_sufficient: false
   - next_action: "human_review"
   - human_review_required: true
   - human_review_reason: "<clear reason for human clarification>"
   - suggested_tool: null, suggested_parameters: null
   - confidence: <= 0.4
6. Reasoning is strictly required: state concisely what evidence has been collected and why this decision was reached. Do not invent facts or assume claims without primary document evidence.

OUTPUT FORMAT:
Respond strictly with a valid JSON object matching this schema:
{{
  "evidence_sufficient": true,
  "reasoning": "<concise explanation of collected evidence and next action>",
  "next_action": "<ONE_OF_ALLOWED_ACTIONS>",
  "suggested_tool": null,
  "suggested_parameters": null,
  "human_review_required": false,
  "human_review_reason": null,
  "confidence": 0.9,
  "decision_source": "LLM"
}}

Respond strictly with valid JSON conforming to this structure."""

        try:
            decision_dict = self.llm.generate_structured(
                prompt=prompt,
                schema_model=SupervisorReplanDecision,
                temperature=0.1,
            )
            decision_dict["decision_source"] = "LLM"
            validated_decision = SupervisorReplanDecision.model_validate(decision_dict)
            if validated_decision.next_action == "verification" and verification_already_run:
                logger.info("Supervisor replan: verification already performed; advancing to response to prevent cycle.")
                return SupervisorReplanDecision(
                    evidence_sufficient=bool(state.get("evidence_sufficient", False)),
                    reasoning=f"Evidence verification already performed (status: {verification_status}). Concluding investigation and proceeding to response.",
                    next_action="response",
                    confidence=validated_decision.confidence,
                    decision_source="HEURISTIC_FALLBACK",
                )

            # Epistemic invariant: relationship evidence must be verified by VerificationAgent before response synthesis
            if validated_decision.next_action == "response" and not verification_already_run and relationship_results:
                logger.info("Supervisor replan: unverified relationship evidence present; enforcing verification before response.")
                return SupervisorReplanDecision(
                    evidence_sufficient=False,
                    reasoning="Retrieved relationship evidence must be verified by VerificationAgent before response synthesis.",
                    next_action="verification",
                    suggested_tool=None,
                    suggested_parameters=None,
                    human_review_required=False,
                    confidence=validated_decision.confidence,
                    decision_source="HEURISTIC_FALLBACK",
                )

            # Document discovery guard: If LLM proposed verification for pure KEYWORD_SEARCH without relationship
            # or document analysis claims, route directly to response to honor the document discovery path
            is_pure_keyword = (
                str(state.get("intent") or "").upper() == "KEYWORD_SEARCH"
                or str((state.get("execution_plan") or {}).get("intent") or "").upper() == "KEYWORD_SEARCH"
            )
            if (
                validated_decision.next_action == "verification"
                and is_pure_keyword
                and has_primary_evidence
                and not relationship_results
                and not document_analysis
            ):
                logger.info("Supervisor replan: pure keyword search with valid search results; advancing directly to response for document discovery.")
                return SupervisorReplanDecision(
                    evidence_sufficient=True,
                    reasoning="Document discovery completed: keyword search successfully retrieved matching Government Orders without unverified claims.",
                    next_action="response",
                    confidence=validated_decision.confidence,
                    decision_source="DETERMINISTIC_RULES",
                )

            return validated_decision
        except Exception as exc:
            logger.warning("LLM replan evaluation fallback: %s", exc)
            is_keyword = (
                str(state.get("intent") or "").upper() == "KEYWORD_SEARCH"
                or str((state.get("execution_plan") or {}).get("intent") or "").upper() == "KEYWORD_SEARCH"
            )
            has_rels = bool(relationship_results)
            has_doc_analysis = bool(document_analysis)

            if is_keyword and has_primary_evidence and not has_rels and not has_doc_analysis:
                return SupervisorReplanDecision(
                    evidence_sufficient=True,
                    reasoning="Document discovery completed: search returned relevant indexed Government Orders.",
                    next_action="response",
                    confidence=0.9,
                    decision_source="HEURISTIC_FALLBACK",
                )

            return SupervisorReplanDecision(
                evidence_sufficient=has_primary_evidence,
                reasoning="Safeguard evaluation: gathered evidence is available for verification.",
                next_action="response" if (has_primary_evidence and verification_already_run) else ("verification" if has_primary_evidence else "search"),
                confidence=0.8 if has_primary_evidence else 0.4,
                decision_source="HEURISTIC_FALLBACK",
            )

    @staticmethod
    def create_trace_item(
        agent: str,
        action: str,
        reasoning: Optional[str] = None,
        tool_called: Optional[str] = None,
        tool_input: Optional[Any] = None,
        tool_output_summary: Optional[str] = None,
        decision: Optional[str] = None,
        decision_source: str = "LLM",
    ) -> Dict[str, Any]:
        """
        Create a standardized trace event dictionary for auditability.
        Safely truncates large inputs/outputs to prevent bloat.
        Explicitly captures decision_source ('LLM' or 'HEURISTIC_FALLBACK').
        """
        def _safe_str(val: Any, max_len: int = 150) -> Optional[str]:
            if val is None:
                return None
            s = str(val)
            return s[:max_len] + "..." if len(s) > max_len else s

        return {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "reasoning": _safe_str(reasoning, 200),
            "tool_called": tool_called,
            "tool_input": _safe_str(tool_input, 100),
            "tool_output_summary": _safe_str(tool_output_summary, 200),
            "decision": decision,
            "decision_source": decision_source,
        }
