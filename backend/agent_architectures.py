"""
agent_architectures.py — Multi-Agent Architecture Router (8 patterns)
Updated to use SQLite memory + Graph of Thoughts. Debate pattern removed —
consensus-by-debate added unnecessary LLM round-trips without adding a
distinct coordination pattern beyond what master_agent/parallel_swarm cover.
"""

import time
from typing import Dict, Any, List, Optional
from schemas import DiagnosticRequest, TaskType
from blackboard import Blackboard
from tracer import AgentTracer
from agent_protocol import (
    create_diagnostic_task, create_safety_task,
    create_cost_task, create_reflection_task
)
import mlflow

ARCH_MASTER_AGENT    = "master_agent"
ARCH_COLLABORATIVE   = "collaborative"
ARCH_SEQUENTIAL      = "sequential_pipeline"
ARCH_PARALLEL_SWARM  = "parallel_swarm"
ARCH_HIERARCHICAL    = "hierarchical"
ARCH_BLACKBOARD_ONLY = "blackboard_only"
ARCH_PEER_TO_PEER    = "peer_to_peer"
ARCH_SELF_IMPROVING  = "self_improving"

ALL_ARCHITECTURES = [
    ARCH_MASTER_AGENT, ARCH_COLLABORATIVE, ARCH_SEQUENTIAL,
    ARCH_PARALLEL_SWARM, ARCH_HIERARCHICAL,
    ARCH_BLACKBOARD_ONLY, ARCH_PEER_TO_PEER, ARCH_SELF_IMPROVING
]


def _init_agents(blackboard, tracer, vector_store):
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.safety_agent import SafetyAgent
    from agents.cost_agent import CostAgent
    from agents.reflection_agent import ReflectionAgent
    from tools import get_diagnostic_tools, get_safety_tools, get_cost_tools
    return {
        "diagnostic": DiagnosticAgent(blackboard, tracer, get_diagnostic_tools(vector_store)),
        "safety":     SafetyAgent(blackboard, tracer, get_safety_tools()),
        "cost":       CostAgent(blackboard, tracer, get_cost_tools()),
        "reflection": ReflectionAgent(blackboard, tracer)
    }


def _build_tasks(request, diag_result=None):
    return {
        "diagnostic": create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or ""),
        "safety":     create_safety_task(request.symptoms, request.fault_codes or "", diag_result or {}),
        "cost":       create_cost_task(likely_causes=(diag_result or {}).get("likely_causes", []), diagnostic_result=diag_result or {})
    }


def _combine(results):
    return {
        "diagnosis":     (results.get("diagnostic") or {}).output if hasattr((results.get("diagnostic") or {}), 'output') else {},
        "safety":        (results.get("safety")     or {}).output if hasattr((results.get("safety")     or {}), 'output') else {},
        "cost_estimate": (results.get("cost")       or {}).output if hasattr((results.get("cost")       or {}), 'output') else {},
        "reflection":    (results.get("reflection") or {}).output if hasattr((results.get("reflection") or {}), 'output') else {}
    }


# ── 1. Master Agent ────────────────────────────────────────

def run_master_agent(request, vector_store):
    from orchestrator import Orchestrator
    orchestrator = Orchestrator(vector_store)
    result = orchestrator.run(request)
    result["architecture_used"]    = ARCH_MASTER_AGENT
    result["pattern_description"]  = "One master orchestrator planned subtasks, delegated to 3 specialists, aggregated with GoT reasoning."
    return result


# ── 2. Collaborative ───────────────────────────────────────

def run_collaborative(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"collab_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    shared = f"SHARED: Vehicle:{request.vehicle_info} Symptoms:{request.symptoms} Codes:{request.fault_codes}"
    results = {}

    for name in ["diagnostic", "safety", "cost"]:
        tasks = _build_tasks(request, results.get("diagnostic", {}) and results["diagnostic"].output)
        result = agents[name].run_with_retry(tasks[name])
        results[name] = result
        shared += f"\n{name.upper()}: {str(result.output)[:150]}"

    reflect_task   = create_reflection_task(_combine(results), request.symptoms, shared)
    results["reflection"] = agents["reflection"].run_with_retry(reflect_task)

    combined = _combine(results)
    combined["architecture_used"]   = ARCH_COLLABORATIVE
    combined["pattern_description"] = "All agents shared full context — each saw previous results."
    return combined


# ── 3. Sequential Pipeline ─────────────────────────────────

def run_sequential_pipeline(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"seq_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    diag_task   = create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or "")
    diag_result = agents["diagnostic"].run_with_retry(diag_task)

    safe_task   = create_safety_task(request.symptoms, request.fault_codes or "", diag_result.output or {}, str(diag_result.output)[:200])
    safe_result = agents["safety"].run_with_retry(safe_task)

    cost_task   = create_cost_task(
        likely_causes=(diag_result.output or {}).get("likely_causes", []),
        diagnostic_result=diag_result.output or {},
        context_summary=str(safe_result.output)[:150]
    )
    cost_result = agents["cost"].run_with_retry(cost_task)

    all_results     = {"diagnostic": diag_result, "safety": safe_result, "cost": cost_result}
    reflect_task    = create_reflection_task(_combine(all_results), request.symptoms)
    reflect_result  = agents["reflection"].run_with_retry(reflect_task)
    all_results["reflection"] = reflect_result

    combined = _combine(all_results)
    combined["architecture_used"]   = ARCH_SEQUENTIAL
    combined["pattern_description"] = "Assembly line: Diagnostic→Safety→Cost→Reflection, each building on previous."
    combined["pipeline_order"]      = ["diagnostic", "safety", "cost", "reflection"]
    return combined


# ── 4. Parallel Swarm ──────────────────────────────────────

def run_parallel_swarm(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"swarm_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    diag_result = agents["diagnostic"].run_with_retry(create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or ""))
    safe_result = agents["safety"].run_with_retry(create_safety_task(request.symptoms, request.fault_codes or ""))
    cost_result = agents["cost"].run_with_retry(create_cost_task(likely_causes=[]))

    all_results    = {"diagnostic": diag_result, "safety": safe_result, "cost": cost_result}
    reflect_task   = create_reflection_task(_combine(all_results), request.symptoms)
    reflect_result = agents["reflection"].run_with_retry(reflect_task)
    all_results["reflection"] = reflect_result

    combined = _combine(all_results)
    combined["architecture_used"]   = ARCH_PARALLEL_SWARM
    combined["pattern_description"] = "All 3 agents ran independently on same input, results merged at end."
    return combined


# ── 5. Hierarchical ────────────────────────────────────────

def run_hierarchical(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"hier_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    symptoms_lower = request.symptoms.lower()
    has_critical   = any(kw in symptoms_lower for kw in ["brake", "airbag", "fire"])
    priority       = "SAFETY_FIRST" if has_critical else "CODE_LOOKUP_FIRST"
    tracer.log_thought("chief_agent", f"Strategy: {priority}")

    diag_result  = agents["diagnostic"].run_with_retry(create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or "", f"Strategy: {priority}"))
    safe_result  = agents["safety"].run_with_retry(create_safety_task(request.symptoms, request.fault_codes or "", diag_result.output or {}, f"Strategy: {priority}"))
    cost_result  = agents["cost"].run_with_retry(create_cost_task(likely_causes=(diag_result.output or {}).get("likely_causes", []), diagnostic_result=diag_result.output or {}))

    all_results    = {"diagnostic": diag_result, "safety": safe_result, "cost": cost_result}
    reflect_result = agents["reflection"].run_with_retry(create_reflection_task(_combine(all_results), request.symptoms))
    all_results["reflection"] = reflect_result

    combined = _combine(all_results)
    combined["architecture_used"]   = ARCH_HIERARCHICAL
    combined["pattern_description"] = f"3-level: Chief({priority}) → Leads(Diag+Safety) → Workers(Cost+Reflect)."
    combined["strategy"]            = {"priority": priority}
    return combined


# ── 7. Blackboard Only ─────────────────────────────────────

def run_blackboard_only(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"bb_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    blackboard.write("user", TaskType.DIAGNOSE, {"symptoms": request.symptoms, "fault_codes": request.fault_codes or "", "status": "PENDING"}, 1.0)

    diag_result  = agents["diagnostic"].run_with_retry(create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or ""))
    blackboard.write("diagnostic_agent", TaskType.DIAGNOSE, diag_result.output or {}, 0.8)

    diag_bb     = blackboard.read(TaskType.DIAGNOSE)
    safe_result = agents["safety"].run_with_retry(create_safety_task(request.symptoms, request.fault_codes or "", diag_bb.content if diag_bb else {}))
    blackboard.write("safety_agent", TaskType.SAFETY, safe_result.output or {}, 0.9)

    cost_result = agents["cost"].run_with_retry(create_cost_task(likely_causes=(diag_result.output or {}).get("likely_causes", [])))
    blackboard.write("cost_agent", TaskType.COST, cost_result.output or {}, 0.75)

    all_results    = {"diagnostic": diag_result, "safety": safe_result, "cost": cost_result}
    reflect_result = agents["reflection"].run_with_retry(create_reflection_task(_combine(all_results), request.symptoms))
    blackboard.write("reflection_agent", TaskType.REFLECT, reflect_result.output or {}, 0.85)

    return {
        "architecture_used":   ARCH_BLACKBOARD_ONLY,
        "pattern_description": "No direct agent comms — all via shared blackboard.",
        "diagnosis":           diag_result.output,
        "safety":              safe_result.output,
        "cost_estimate":       cost_result.output,
        "reflection":          reflect_result.output,
        "blackboard_log":      blackboard.get_audit_log()
    }


# ── 8. Peer to Peer ────────────────────────────────────────

def run_peer_to_peer(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"p2p_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    diag_result  = agents["diagnostic"].run_with_retry(create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or ""))
    safe_result  = agents["safety"].run_with_retry(create_safety_task(request.symptoms, request.fault_codes or "", diag_result.output or {}, f"From diagnostic peer: {str(diag_result.output)[:150]}"))
    cost_result  = agents["cost"].run_with_retry(create_cost_task(likely_causes=(diag_result.output or {}).get("likely_causes", []), context_summary=f"From safety peer: {str(safe_result.output)[:150]}"))

    all_results    = {"diagnostic": diag_result, "safety": safe_result, "cost": cost_result}
    reflect_result = agents["reflection"].run_with_retry(create_reflection_task(_combine(all_results), request.symptoms, "P2P chain complete"))
    all_results["reflection"] = reflect_result

    combined = _combine(all_results)
    combined["architecture_used"]       = ARCH_PEER_TO_PEER
    combined["pattern_description"]     = "No orchestrator — agents communicate directly peer-to-peer."
    combined["communication_path"]      = "diagnostic → safety → cost → reflection"
    return combined


# ── 9. Self Improving ──────────────────────────────────────

def run_self_improving(request, vector_store):
    blackboard = Blackboard()
    tracer     = AgentTracer(f"self_{int(time.time())}")
    agents     = _init_agents(blackboard, tracer, vector_store)

    MAX_ROUNDS  = 3
    THRESHOLD   = 0.7
    improvement_log = []
    current_diagnosis = None

    for round_num in range(MAX_ROUNDS):
        context = f"Previous attempt: {str(current_diagnosis)[:200]}. Improve." if current_diagnosis else ""
        diag_task   = create_diagnostic_task(request.symptoms, request.fault_codes or "", request.vehicle_info or "", context)
        diag_result = agents["diagnostic"].run_with_retry(diag_task)

        # Self-critique
        primary  = str((diag_result.output or {}).get("primary_issue", "")).lower()
        causes   = (diag_result.output or {}).get("likely_causes", [])
        score    = 0.0
        if len(primary) > 20:   score += 0.4
        if len(causes) >= 2:    score += 0.3
        if any(kw in primary for kw in ["spark", "sensor", "valve", "pump"]): score += 0.3

        improvement_log.append({"round": round_num + 1, "score": round(score, 2), "output": str(diag_result.output)[:150]})
        current_diagnosis = diag_result.output

        if score >= THRESHOLD:
            break

    safe_result  = agents["safety"].run_with_retry(create_safety_task(request.symptoms, request.fault_codes or ""))
    cost_result  = agents["cost"].run_with_retry(create_cost_task(likely_causes=(current_diagnosis or {}).get("likely_causes", [])))

    return {
        "architecture_used":   ARCH_SELF_IMPROVING,
        "pattern_description": f"Single agent ran {len(improvement_log)} self-improvement rounds until satisfied.",
        "diagnosis":           current_diagnosis or {},
        "safety":              safe_result.output,
        "cost_estimate":       cost_result.output,
        "improvement_log":     improvement_log,
        "rounds_taken":        len(improvement_log)
    }


# ── Main Router ────────────────────────────────────────────

def run(request, vector_store, arch_kind: str = ARCH_MASTER_AGENT) -> Dict[str, Any]:
    start = time.time()

    arch_map = {
        ARCH_MASTER_AGENT:    run_master_agent,
        ARCH_COLLABORATIVE:   run_collaborative,
        ARCH_SEQUENTIAL:      run_sequential_pipeline,
        ARCH_PARALLEL_SWARM:  run_parallel_swarm,
        ARCH_HIERARCHICAL:    run_hierarchical,
        ARCH_BLACKBOARD_ONLY: run_blackboard_only,
        ARCH_PEER_TO_PEER:    run_peer_to_peer,
        ARCH_SELF_IMPROVING:  run_self_improving
    }

    if arch_kind not in arch_map:
        raise ValueError(f"Unknown architecture: '{arch_kind}'. Choose from: {ALL_ARCHITECTURES}")

    result = arch_map[arch_kind](request, vector_store)
    result["total_time"] = f"{time.time() - start:.2f}s"

    with mlflow.start_run(run_name=f"arch_{arch_kind}"):
        mlflow.log_param("architecture", arch_kind)
        mlflow.log_param("symptoms",     request.symptoms[:100])
        mlflow.log_metric("total_time",  time.time() - start)

    return result


def compare_architectures(request, vector_store, arch_list=None) -> Dict[str, Any]:
    if arch_list is None:
        arch_list = ALL_ARCHITECTURES
    comparison = {}
    for arch in arch_list:
        try:
            result = run(request, vector_store, arch)
            comparison[arch] = {"status": "success", "total_time": result.get("total_time", "N/A"), "has_diagnosis": bool(result.get("diagnosis"))}
        except Exception as e:
            comparison[arch] = {"status": "error", "error": str(e)}
    return {
        "request_symptoms":  request.symptoms,
        "architectures_run": len(arch_list),
        "comparison":        comparison,
        "recommendation":    _recommend_architecture(request)
    }


def _recommend_architecture(request) -> str:
    symptoms_lower = request.symptoms.lower()
    if any(kw in symptoms_lower for kw in ["brake", "airbag", "fire"]):
        return ARCH_MASTER_AGENT
    if len(request.symptoms.split()) < 5:
        return ARCH_PARALLEL_SWARM
    if request.fault_codes:
        return ARCH_SEQUENTIAL
    if len(request.symptoms.split()) > 20:
        return ARCH_HIERARCHICAL
    return ARCH_MASTER_AGENT


def list_architectures() -> Dict:
    return {
        ARCH_MASTER_AGENT:    "One orchestrator delegates to specialist workers, aggregates with GoT",
        ARCH_COLLABORATIVE:   "All agents share full context",
        ARCH_SEQUENTIAL:      "Assembly line A→B→C→D",
        ARCH_PARALLEL_SWARM:  "All independent, results merged",
        ARCH_HIERARCHICAL:    "3-level: Chief → Leads → Workers",
        ARCH_BLACKBOARD_ONLY: "All via shared blackboard",
        ARCH_PEER_TO_PEER:    "No orchestrator, direct comms",
        ARCH_SELF_IMPROVING:  "Single agent iterates with self-critique"
    }