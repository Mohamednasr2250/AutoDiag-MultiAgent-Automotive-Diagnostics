"""
orchestrator.py — Master Agent
Instantiated once at app startup as Orchestrator(vector_store), reused
across requests. Each .run() call builds a fresh Blackboard/Tracer/Memory
scoped to that session.
"""

import time
from typing import Dict, Any, Optional

from blackboard import Blackboard
from tracer import AgentTracer
from schemas import TaskType, DiagnosticRequest
from agent_protocol import (
    create_diagnostic_task, create_safety_task, create_cost_task, create_reflection_task
)
from memory_manager import MemoryManager
from reasoning_engine import GraphOfThoughts, WorldModel
from guardrails import needs_human_review, build_human_review_response
from tools import get_diagnostic_tools, get_safety_tools, get_cost_tools

from agents.diagnostic_agent import DiagnosticAgent
from agents.safety_agent import SafetyAgent
from agents.cost_agent import CostAgent
from agents.reflection_agent import ReflectionAgent


class Orchestrator:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def resolve_conflicts(self, results: Dict[str, Any]) -> Dict[str, Any]:
        safety = results.get("safety", {})
        if safety.get("risk_level") == "CRITICAL":
            results["conflict_resolution"] = "Safety agent CRITICAL finding takes precedence over all other recommendations."
            results["override_priority"] = "safety"
        else:
            results["conflict_resolution"] = "No conflicts — all agents in agreement."
            results["override_priority"] = None
        return results

    def run(self, request: DiagnosticRequest, session_id: Optional[str] = None, use_got: bool = False) -> Dict[str, Any]:
        start = time.time()
        session_id = session_id or f"sess_{int(start)}"

        blackboard = Blackboard()
        tracer = AgentTracer(session_id)
        memory = MemoryManager(session_id)

        diagnostic_agent = DiagnosticAgent(blackboard, tracer, get_diagnostic_tools(self.vector_store))
        safety_agent = SafetyAgent(blackboard, tracer, get_safety_tools())
        cost_agent = CostAgent(blackboard, tracer, get_cost_tools())
        reflection_agent = ReflectionAgent(blackboard, tracer)

        symptoms = request.symptoms
        fault_codes = request.fault_codes or ""
        vehicle_info = request.vehicle_info or ""

        memory.add_turn("user", symptoms)
        context_summary = memory.build_memory_context(symptoms)

        got_result = None
        if use_got:
            got = GraphOfThoughts(diagnostic_agent.llm, breadth=3, depth=2)
            got_result = got.run(f"Symptoms: {symptoms}. Fault codes: {fault_codes}.")
            if got_result.get("best_hypothesis"):
                context_summary = (context_summary + f"\n\nGraph-of-Thoughts hypothesis: "
                                    f"{got_result['best_hypothesis'][:200]}").strip()

        diag_task = create_diagnostic_task(symptoms, fault_codes, vehicle_info, context_summary)
        diag_result = diagnostic_agent.run_with_retry(diag_task)
        blackboard.write("diagnostic_agent", TaskType.DIAGNOSE, diag_result.output, diag_result.confidence)

        safety_task = create_safety_task(symptoms, fault_codes, diag_result.output, context_summary)
        safety_result = safety_agent.run_with_retry(safety_task)
        blackboard.write("safety_agent", TaskType.SAFETY, safety_result.output, safety_result.confidence)

        likely_causes = diag_result.output.get("likely_causes", [])
        cost_task = create_cost_task(likely_causes, diag_result.output, context_summary)
        cost_result = cost_agent.run_with_retry(cost_task)
        blackboard.write("cost_agent", TaskType.COST, cost_result.output, cost_result.confidence)

        combined = blackboard.read_all()
        combined_dict = {k: v.content for k, v in combined.items()}
        reflect_task = create_reflection_task(combined_dict, symptoms, context_summary)
        reflect_result = reflection_agent.run_with_retry(reflect_task)
        blackboard.write("reflection_agent", TaskType.REFLECT, reflect_result.output, reflect_result.confidence)

        final_report = {
            "diagnosis": diag_result.output,
            "safety": safety_result.output,
            "cost_estimate": cost_result.output,
            "reflection": reflect_result.output,
        }
        final_report = self.resolve_conflicts(final_report)

        if use_got and got_result:
            final_report["graph_of_thoughts"] = got_result

        primary_rec = reflect_result.output.get("final_recommendation") or diag_result.output.get("primary_issue", "")
        if primary_rec:
            world_model = WorldModel(diagnostic_agent.llm)
            final_report["world_model_validation"] = world_model.validate_and_revise(str(primary_rec))

        needs_human, reason = needs_human_review(str(diag_result.output), safety_result.output, fault_codes)
        final_report["needs_human_review"] = needs_human
        if needs_human:
            final_report["human_review"] = build_human_review_response(reason, final_report)

        final_report["session_id"] = session_id
        final_report["is_safety_critical"] = safety_result.output.get("risk_level") == "CRITICAL"

        memory.add_turn("assistant", str(final_report.get("reflection", {})))
        memory.save_diagnosis(symptoms, str(final_report["diagnosis"]), "pending_feedback", rating=True)

        tracer.log_final_answer("orchestrator", "Diagnosis complete")
        final_report["trace_summary"] = tracer.get_summary()
        tracer.flush_to_mlflow()

        final_report["total_time"] = f"{time.time() - start:.2f}s"
        return final_report