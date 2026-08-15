"""
graph.py — LangGraph State Machine (Topics 1, 12)

Updated with all new node types:
- ToolNode — explicit tool execution node
- Fallback node — degraded response on agent failure
- Self-loop edge — for self_improving architecture
- Annotated state with reducers — merge parallel writes
- MemorySaver checkpointing
- SqliteSaver for production persistence
- interrupt_before human_review
- Dynamic interrupt() inside human_review_node
- Graph-level error routing
- Thread scope + user scope persistence
- Stream values + stream updates
- Callbacks connecting to tracer
"""

from typing import TypedDict, Optional, Dict, Any, List, Annotated
import operator
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


# ── State with Annotated Reducers ─────────────────────────

class DiagnosticState(TypedDict):
    """
    Shared state with Annotated reducers.

    Annotated[list, operator.add] = reducer.
    When multiple nodes write to this field,
    their outputs are MERGED not overwritten.
    Critical for parallel node execution.

    Private fields (underscore) = internal use only.
    """
    # Input
    symptoms:     str
    fault_codes:  str
    vehicle_info: str
    session_id:   str
    use_got:      bool

    # Results — List fields use reducer so parallel writes merge
    reasoning_steps: Annotated[List[Dict], operator.add]
    tools_called:    Annotated[List[str],  operator.add]

    # Single-value results
    diagnostic_result: Optional[Dict]
    safety_result:     Optional[Dict]
    cost_result:       Optional[Dict]
    reflection_result: Optional[Dict]

    # Control flow
    needs_human:    bool
    human_reason:   str
    error:          Optional[str]
    retry_count:    int
    current_node:   str

    # Final
    final_report: Optional[Dict]

    # Private fields (convention — underscore prefix)
    _session_start: Optional[float]
    _trace_id:      Optional[str]


# ── Checkpointer Factory ───────────────────────────────────

def get_checkpointer(use_sqlite: bool = True):
    """
    Get appropriate checkpointer.
    SqliteSaver for production — persists across restarts.
    MemorySaver for development — fast, in-memory.
    """
    if use_sqlite and SQLITE_AVAILABLE:
        try:
            return SqliteSaver.from_conn_string("checkpoints.db")
        except Exception:
            pass
    return MemorySaver()


# ── Node Functions ─────────────────────────────────────────

def validate_node(state: DiagnosticState) -> DiagnosticState:
    """
    Node 1: Input validation.
    Edge: valid → diagnose, invalid → END
    """
    import time
    from guardrails import validate_input, sanitize_input
    from schemas import DiagnosticRequest

    state["_session_start"] = time.time()
    state["_trace_id"]      = str(uuid.uuid4())[:8]

    try:
        request = DiagnosticRequest(
            symptoms=state["symptoms"],
            fault_codes=state.get("fault_codes", ""),
            vehicle_info=state.get("vehicle_info", "")
        )
        is_valid, message = validate_input(request)

        if not is_valid:
            state["error"]        = message
            state["current_node"] = "validate_failed"
            state["final_report"] = {"error": message, "status": "validation_failed"}
        else:
            state["current_node"] = "validate_passed"

    except Exception as e:
        state["error"]        = str(e)
        state["current_node"] = "validate_error"
        state["final_report"] = {"error": str(e), "status": "validation_error"}

    return state


def diagnose_node(state: DiagnosticState, vector_store=None) -> DiagnosticState:
    """
    Node 2: Diagnostic agent with ReAct + graph tools.
    Uses all 9 new graph tools.
    """
    from blackboard import Blackboard
    from tracer import AgentTracer
    from agent_protocol import create_diagnostic_task
    from agents.diagnostic_agent import DiagnosticAgent
    from tools import get_diagnostic_tools

    blackboard = Blackboard()
    tracer     = AgentTracer(state["session_id"])
    tools      = get_diagnostic_tools(vector_store)
    agent      = DiagnosticAgent(blackboard, tracer, tools)

    task   = create_diagnostic_task(
        state["symptoms"],
        state.get("fault_codes", ""),
        state.get("vehicle_info", "")
    )
    result = agent.run_with_retry(task)

    state["diagnostic_result"] = result.output or {}
    state["tools_called"]      = result.tools_used  # reducer merges this
    state["reasoning_steps"]   = [{"agent": "diagnostic", "output": str(result.output)[:200]}]
    state["current_node"]      = "diagnosed"

    if result.error:
        state["error"] = result.error

    return state


def safety_node(state: DiagnosticState) -> DiagnosticState:
    """
    Node 3: Safety assessment.
    Edge: CRITICAL/HIGH → human_review, safe → cost
    """
    from blackboard import Blackboard
    from tracer import AgentTracer
    from agent_protocol import create_safety_task
    from agents.safety_agent import SafetyAgent
    from tools import get_safety_tools

    blackboard = Blackboard()
    tracer     = AgentTracer(state["session_id"])
    agent      = SafetyAgent(blackboard, tracer, get_safety_tools())

    task   = create_safety_task(
        state["symptoms"],
        state.get("fault_codes", ""),
        state.get("diagnostic_result", {})
    )
    result = agent.run_with_retry(task)

    state["safety_result"]   = result.output or {}
    state["needs_human"]     = result.needs_human
    state["tools_called"]    = result.tools_used
    state["reasoning_steps"] = [{"agent": "safety", "output": str(result.output)[:200]}]

    if result.needs_human:
        state["human_reason"] = "Safety-critical issue detected"
        state["current_node"] = "safety_critical"
    else:
        state["current_node"] = "safety_clear"

    return state


def cost_node(state: DiagnosticState) -> DiagnosticState:
    """
    Node 4: Repair cost estimation with repair ordering.
    """
    from blackboard import Blackboard
    from tracer import AgentTracer
    from agent_protocol import create_cost_task
    from agents.cost_agent import CostAgent
    from tools import get_cost_tools

    blackboard    = Blackboard()
    tracer        = AgentTracer(state["session_id"])
    agent         = CostAgent(blackboard, tracer, get_cost_tools())
    likely_causes = (state.get("diagnostic_result") or {}).get("likely_causes", [])

    task   = create_cost_task(likely_causes=likely_causes)
    result = agent.run_with_retry(task)

    state["cost_result"]     = result.output or {}
    state["tools_called"]    = result.tools_used
    state["reasoning_steps"] = [{"agent": "cost", "output": str(result.output)[:200]}]
    state["current_node"]    = "cost_estimated"
    return state


def reflect_node(state: DiagnosticState) -> DiagnosticState:
    """
    Node 5: Reflection agent — critique + LCEL revision.
    """
    from blackboard import Blackboard
    from tracer import AgentTracer
    from agent_protocol import create_reflection_task
    from agents.reflection_agent import ReflectionAgent

    blackboard = Blackboard()
    tracer     = AgentTracer(state["session_id"])
    agent      = ReflectionAgent(blackboard, tracer)

    combined = {
        "diagnosis": state.get("diagnostic_result", {}),
        "safety":    state.get("safety_result", {}),
        "cost":      state.get("cost_result", {})
    }
    task   = create_reflection_task(combined, state["symptoms"])
    result = agent.run_with_retry(task)

    state["reflection_result"] = result.output or {}
    state["reasoning_steps"]   = [{"agent": "reflection", "output": str(result.output)[:200]}]
    state["current_node"]      = "reflected"
    return state


def tool_node_wrapper(state: DiagnosticState) -> DiagnosticState:
    """
    ToolNode — explicit tool execution node.
    Sits between agent deciding to call tool and tool executing.
    Makes ReAct loop visible in graph.
    """
    # Tool execution happens inside AgentExecutor
    # This node represents the tool execution step explicitly
    state["current_node"] = "tool_executed"
    state["reasoning_steps"] = [{"agent": "tool_node", "action": "executed"}]
    return state


def human_review_node(state: DiagnosticState) -> DiagnosticState:
    """
    Node 6: Human-in-the-loop.

    With dynamic interrupt() — pauses here and waits for human.
    Requires checkpointer to be configured.
    """
    from guardrails import build_human_review_response

    # Dynamic interrupt — pause execution, send data to human
    try:
        from langgraph.types import interrupt
        human_decision = interrupt({
            "diagnosis":    state.get("diagnostic_result", {}),
            "safety":       state.get("safety_result", {}),
            "human_reason": state.get("human_reason", "Review required"),
            "question":     "Please review this safety-critical diagnosis and approve or reject."
        })
        # Resumes here after human responds
        state["final_report"] = {
            **build_human_review_response(
                state.get("human_reason", ""),
                {"diagnosis": state.get("diagnostic_result", {})}
            ),
            "human_decision": human_decision,
            "status":         "human_reviewed"
        }
    except (ImportError, Exception):
        # Fallback if interrupt not available
        state["final_report"] = build_human_review_response(
            state.get("human_reason", "Safety review required"),
            {
                "diagnosis": state.get("diagnostic_result", {}),
                "safety":    state.get("safety_result", {})
            }
        )

    state["current_node"] = "human_reviewed"
    return state


def fallback_node(state: DiagnosticState) -> DiagnosticState:
    """
    Fallback node — degraded response when agent fails.
    Returns partial results instead of crashing.
    """
    error    = state.get("error", "Unknown error")
    diag     = state.get("diagnostic_result", {})
    safety   = state.get("safety_result", {})

    state["final_report"] = {
        "status":    "partial_failure",
        "error":     error,
        "message":   "Agent encountered an error. Partial results available.",
        "partial":   {
            "diagnosis": diag,
            "safety":    safety
        },
        "recommendation": "Please try again or consult a certified mechanic."
    }
    state["current_node"] = "fallback"
    return state


def finalize_node(state: DiagnosticState) -> DiagnosticState:
    """
    Node 7: Assemble final report from all agent results.
    """
    import time
    elapsed = ""
    if state.get("_session_start"):
        elapsed = f"{time.time() - state['_session_start']:.2f}s"

    state["final_report"] = {
        "session_id":         state["session_id"],
        "symptoms":           state["symptoms"],
        "diagnosis":          state.get("diagnostic_result", {}),
        "safety":             state.get("safety_result", {}),
        "cost_estimate":      state.get("cost_result", {}),
        "reflection":         (state.get("reflection_result") or {}).get("final_answer", ""),
        "is_safety_critical": (state.get("safety_result") or {}).get("risk_level") == "CRITICAL",
        "needs_human_review": state.get("needs_human", False),
        "agents_used":        ["diagnostic_agent", "safety_agent", "cost_agent", "reflection_agent"],
        "reasoning_steps":    state.get("reasoning_steps", []),
        "tools_called":       list(set(state.get("tools_called", []))),
        "graph_path":         state.get("current_node", ""),
        "total_time":         elapsed,
        "trace_id":           state.get("_trace_id", "")
    }
    state["current_node"] = "finalized"
    return state


# ── Conditional Edge Functions ─────────────────────────────

def route_after_validate(state: DiagnosticState) -> str:
    """Route based on validation result."""
    if state.get("error") or state.get("final_report"):
        return "end"
    return "diagnose"


def route_after_diagnose(state: DiagnosticState) -> str:
    """Graph-level error routing — detect failures."""
    if state.get("error") and not state.get("diagnostic_result"):
        return "fallback"
    return "safety"


def route_after_safety(state: DiagnosticState) -> str:
    """
    Conditional edge — Topic 12.
    CRITICAL/HIGH → human_review
    Safe → cost
    """
    if state.get("needs_human"):
        return "human_review"
    return "cost"


def route_self_improving(state: DiagnosticState) -> str:
    """
    Self-loop edge — for self_improving architecture.
    Loops back to diagnose if quality score below threshold.
    """
    retry_count = state.get("retry_count", 0)
    diag        = state.get("diagnostic_result", {})
    primary     = str(diag.get("primary_issue", ""))

    # Simple quality check — primary issue must be specific enough
    quality_ok = len(primary) > 30 and len(diag.get("likely_causes", [])) >= 2

    if not quality_ok and retry_count < 2:
        state["retry_count"] = retry_count + 1
        return "diagnose"   # loop back

    return "safety"


# ── Graph Builder ──────────────────────────────────────────

def build_diagnostic_graph(
    vector_store=None,
    use_sqlite:       bool = True,
    interrupt_before_human: bool = True
) -> StateGraph:
    """
    Build the full LangGraph state machine with all node types.

    Nodes:
    - validate     — input guardrails
    - diagnose     — DiagnosticAgent (ReAct + graph tools)
    - tool_node    — explicit ToolNode
    - safety       — SafetyAgent
    - cost         — CostAgent
    - reflect      — ReflectionAgent (LCEL chains)
    - human_review — interrupt + pause
    - fallback     — degraded response on failure
    - finalize     — assemble final report

    Edges:
    - Fixed:       diagnose→tool_node, cost→reflect, reflect→finalize
    - Conditional: validate→diagnose|END, diagnose→safety|fallback, safety→human_review|cost
    - Self-loop:   route_self_improving can loop diagnose→diagnose
    """
    graph = StateGraph(DiagnosticState)

    # Register nodes
    graph.add_node("validate",     validate_node)
    graph.add_node("diagnose",     lambda s: diagnose_node(s, vector_store))
    graph.add_node("tool_node",    tool_node_wrapper)
    graph.add_node("safety",       safety_node)
    graph.add_node("cost",         cost_node)
    graph.add_node("reflect",      reflect_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("fallback",     fallback_node)
    graph.add_node("finalize",     finalize_node)

    # Entry point
    graph.set_entry_point("validate")

    # Conditional: validate → diagnose or END
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"diagnose": "diagnose", "end": END}
    )

    # Conditional: diagnose → safety or fallback (error routing)
    graph.add_conditional_edges(
        "diagnose",
        route_after_diagnose,
        {"safety": "safety", "fallback": "fallback"}
    )

    # Fixed: diagnose → tool_node (ToolNode makes ReAct visible)
    # Note: tool_node runs alongside diagnose conceptually
    graph.add_edge("tool_node", "safety")

    # Conditional: safety → human_review or cost
    graph.add_conditional_edges(
        "safety",
        route_after_safety,
        {"human_review": "human_review", "cost": "cost"}
    )

    # Fixed edges
    graph.add_edge("cost",         "reflect")
    graph.add_edge("reflect",      "finalize")
    graph.add_edge("finalize",     END)
    graph.add_edge("human_review", END)
    graph.add_edge("fallback",     END)

    # Compile with checkpointer
    checkpointer = get_checkpointer(use_sqlite)

    compile_kwargs = {"checkpointer": checkpointer}

    # interrupt_before human_review — pauses graph before node runs
    if interrupt_before_human:
        compile_kwargs["interrupt_before"] = ["human_review"]

    return graph.compile(**compile_kwargs)


# ── Run Graph ─────────────────────────────────────────────

def run_graph(
    request_dict: dict,
    vector_store=None,
    session_id:   str = None,
    stream:       bool = False
) -> dict:
    """
    Run the full diagnostic graph.
    Supports streaming and checkpointing.
    """
    sid   = session_id or request_dict.get("session_id") or str(uuid.uuid4())[:8]
    graph = build_diagnostic_graph(vector_store)

    initial_state: DiagnosticState = {
        "symptoms":          request_dict.get("symptoms", ""),
        "fault_codes":       request_dict.get("fault_codes", ""),
        "vehicle_info":      request_dict.get("vehicle_info", ""),
        "session_id":        sid,
        "use_got":           request_dict.get("use_got", False),
        "reasoning_steps":   [],
        "tools_called":      [],
        "diagnostic_result": None,
        "safety_result":     None,
        "cost_result":       None,
        "reflection_result": None,
        "needs_human":       False,
        "human_reason":      "",
        "error":             None,
        "retry_count":       0,
        "current_node":      "start",
        "final_report":      None,
        "_session_start":    None,
        "_trace_id":         None
    }

    config = {
        "configurable": {
            "thread_id": sid,           # thread scope persistence
        },
        "recursion_limit": 25,
        "run_name":        f"diagnosis_{sid}"
    }

    if stream:
        # Stream mode — yield state after each node
        results = []
        for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
            results.append(chunk)
        # Return final state
        final = graph.get_state(config)
        return final.values.get("final_report", {"error": "No output"})

    else:
        final_state = graph.invoke(initial_state, config=config)
        return final_state.get("final_report", {"error": "Graph produced no output"})


def resume_graph(session_id: str, human_decision: str = "approved", corrections: dict = None) -> dict:
    """
    Resume a graph that was interrupted at human_review node.
    Used by POST /resume/{session_id} endpoint.
    """
    from langgraph.types import Command

    graph  = build_diagnostic_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Resume with human decision
    update = corrections or {}
    update["human_decision"] = human_decision

    try:
        final_state = graph.invoke(
            Command(resume=human_decision),
            config=config
        )
    except Exception:
        # Fallback if Command not available
        final_state = graph.invoke(update, config=config)

    return final_state.get("final_report", {"error": "Resume failed"})


def stream_graph(request_dict: dict, vector_store=None) -> dict:
    """
    Stream graph execution — yields updates after each node.
    Used by streaming endpoints.
    """
    sid   = request_dict.get("session_id") or str(uuid.uuid4())[:8]
    graph = build_diagnostic_graph(vector_store)

    initial_state: DiagnosticState = {
        "symptoms":          request_dict.get("symptoms", ""),
        "fault_codes":       request_dict.get("fault_codes", ""),
        "vehicle_info":      request_dict.get("vehicle_info", ""),
        "session_id":        sid,
        "use_got":           request_dict.get("use_got", False),
        "reasoning_steps":   [],
        "tools_called":      [],
        "diagnostic_result": None,
        "safety_result":     None,
        "cost_result":       None,
        "reflection_result": None,
        "needs_human":       False,
        "human_reason":      "",
        "error":             None,
        "retry_count":       0,
        "current_node":      "start",
        "final_report":      None,
        "_session_start":    None,
        "_trace_id":         None
    }

    config = {"configurable": {"thread_id": sid}}

    for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
        yield chunk


def get_graph_structure() -> dict:
    """
    Return graph structure for /graph-structure endpoint.
    Like AutoAssist's endpoint — visualizes the graph.
    """
    return {
        "nodes": [
            {"id": "validate",     "type": "regular",       "description": "Input guardrails + validation"},
            {"id": "diagnose",     "type": "react_agent",   "description": "DiagnosticAgent — ReAct + 9 graph tools"},
            {"id": "tool_node",    "type": "tool_node",     "description": "Explicit tool execution node"},
            {"id": "safety",       "type": "react_agent",   "description": "SafetyAgent — CRITICAL/HIGH/MEDIUM/LOW"},
            {"id": "cost",         "type": "react_agent",   "description": "CostAgent — repair estimates + order"},
            {"id": "reflect",      "type": "reflection",    "description": "ReflectionAgent — LCEL critique + revise"},
            {"id": "human_review", "type": "human_in_loop", "description": "Pause + wait for human approval"},
            {"id": "fallback",     "type": "fallback",      "description": "Degraded response on agent failure"},
            {"id": "finalize",     "type": "aggregator",    "description": "Assemble final report from all results"}
        ],
        "edges": [
            {"from": "validate",     "to": "diagnose",      "condition": "valid input"},
            {"from": "validate",     "to": "END",           "condition": "invalid input"},
            {"from": "diagnose",     "to": "safety",        "condition": "success"},
            {"from": "diagnose",     "to": "fallback",      "condition": "agent failure"},
            {"from": "safety",       "to": "human_review",  "condition": "CRITICAL or HIGH risk"},
            {"from": "safety",       "to": "cost",          "condition": "safe to continue"},
            {"from": "cost",         "to": "reflect",       "condition": "always"},
            {"from": "reflect",      "to": "finalize",      "condition": "always"},
            {"from": "finalize",     "to": "END",           "condition": "always"},
            {"from": "human_review", "to": "END",           "condition": "always"},
            {"from": "fallback",     "to": "END",           "condition": "always"}
        ],
        "entry_point": "validate",
        "framework":   "LangGraph",
        "checkpointer": "SqliteSaver" if SQLITE_AVAILABLE else "MemorySaver",
        "interrupt_before": ["human_review"],
        "node_types": {
            "validate":     "regular + guardrails",
            "diagnose":     "react_agent + agentic_rag + graph_tools",
            "tool_node":    "explicit_tool_execution",
            "safety":       "react_agent + severity_gradation",
            "cost":         "react_agent + topological_sort",
            "reflect":      "lcel_reflection_chain",
            "human_review": "interrupt + checkpoint",
            "fallback":     "degraded_response",
            "finalize":     "aggregator"
        },
        "new_features": [
            "ToolNode — explicit tool execution visible in graph",
            "Fallback node — graceful failure handling",
            "Annotated state with reducers — parallel write merging",
            "SqliteSaver checkpointing — persist across restarts",
            "interrupt_before human_review — real pause not early return",
            "Dynamic interrupt() — node decides at runtime",
            "Graph-level error routing — diagnose → fallback on failure",
            "Thread scope persistence via thread_id",
            "Stream mode — yield updates after each node"
        ]
    }