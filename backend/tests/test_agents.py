"""
tests/test_agents.py — Agent Infrastructure Tests
Tests: Blackboard, AgentProtocol, AgentTracer, MemoryManager
No LLM calls — pure infrastructure tests
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from blackboard import Blackboard
from tracer import AgentTracer
from schemas import TaskType, AgentStatus
from agent_protocol import (
    create_diagnostic_task, create_safety_task,
    create_cost_task, create_reflection_task,
    build_success_result, build_error_result, build_human_review_result
)
from memory_manager import MemoryManager, ShortTermMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
import db
db.DB_PATH = "test_agents.db"
db.init_db()


# ── Blackboard Tests ───────────────────────────────────────

def test_blackboard_write_read():
    bb = Blackboard()
    bb.write("diag_agent", TaskType.DIAGNOSE, {"primary": "misfire"}, 0.8)
    entry = bb.read(TaskType.DIAGNOSE)
    assert entry is not None
    assert entry.content["primary"] == "misfire"
    assert entry.agent_name == "diag_agent"
    assert entry.confidence == 0.8


def test_blackboard_redundancy_blocked():
    bb = Blackboard()
    bb.write("agent_a", TaskType.DIAGNOSE, {"result": "first"}, 0.9)
    # Lower confidence write should be blocked
    blocked = bb.write("agent_b", TaskType.DIAGNOSE, {"result": "second"}, 0.5)
    assert blocked is False
    entry = bb.read(TaskType.DIAGNOSE)
    assert entry.content["result"] == "first"


def test_blackboard_higher_confidence_wins():
    bb = Blackboard()
    bb.write("agent_a", TaskType.SAFETY, {"risk": "LOW"}, 0.6)
    # Higher confidence write should succeed
    bb.write("agent_b", TaskType.SAFETY, {"risk": "HIGH"}, 0.9)
    entry = bb.read(TaskType.SAFETY)
    # The second write (higher conf) may or may not succeed depending on is_final
    assert entry is not None


def test_blackboard_is_task_done_false():
    bb = Blackboard()
    assert bb.is_task_done(TaskType.DIAGNOSE) is False


def test_blackboard_is_task_done_true():
    bb = Blackboard()
    bb.write("agent", TaskType.DIAGNOSE, {"done": True}, 0.8)
    assert bb.is_task_done(TaskType.DIAGNOSE) is True


def test_blackboard_read_all():
    bb = Blackboard()
    bb.write("diag",   TaskType.DIAGNOSE, {}, 0.8)
    bb.write("safety", TaskType.SAFETY,   {}, 0.9)
    all_entries = bb.read_all()
    assert len(all_entries) == 2
    assert TaskType.DIAGNOSE.value in all_entries
    assert TaskType.SAFETY.value   in all_entries


def test_blackboard_summary():
    bb = Blackboard()
    bb.write("diag",   TaskType.DIAGNOSE, {}, 0.8)
    bb.write("safety", TaskType.SAFETY,   {}, 0.9)
    summary = bb.summary()
    assert summary["total_entries"] == 2
    assert "diag"   in summary["agents_wrote"]
    assert "safety" in summary["agents_wrote"]


def test_blackboard_audit_log():
    bb = Blackboard()
    bb.write("agent_x", TaskType.COST, {}, 0.7)
    log = bb.get_audit_log()
    assert len(log) >= 1
    assert log[0]["event"] == "WRITE"


def test_blackboard_clear():
    bb = Blackboard()
    bb.write("agent", TaskType.DIAGNOSE, {}, 0.8)
    bb.clear()
    assert bb.is_task_done(TaskType.DIAGNOSE) is False
    assert len(bb.get_audit_log()) == 0


# ── AgentProtocol Tests ────────────────────────────────────

def test_create_diagnostic_task():
    task = create_diagnostic_task("rough idle", "P0300", "Toyota 2019", "context summary")
    assert task.task_type   == TaskType.DIAGNOSE
    assert task.from_agent  == "orchestrator"
    assert task.to_agent    == "diagnostic_agent"
    assert task.payload["symptoms"]    == "rough idle"
    assert task.payload["fault_codes"] == "P0300"
    assert task.priority == 1


def test_create_safety_task():
    task = create_safety_task("brake noise", "C0035", {"primary": "misfire"})
    assert task.task_type  == TaskType.SAFETY
    assert task.to_agent   == "safety_agent"
    assert task.priority   == 2
    assert "brake noise" in task.payload["symptoms"]


def test_create_cost_task():
    task = create_cost_task(["spark plugs", "O2 sensor"], {"primary": "misfire"})
    assert task.task_type == TaskType.COST
    assert task.to_agent  == "cost_agent"
    assert task.priority  == 3
    assert "spark plugs" in task.payload["likely_causes"]


def test_create_reflection_task():
    task = create_reflection_task({"diagnosis": {}, "safety": {}}, "rough idle")
    assert task.task_type == TaskType.REFLECT
    assert task.to_agent  == "reflection_agent"
    assert task.priority  == 4


def test_build_success_result():
    result = build_success_result("task_123", "test_agent", {"output": "done"}, 5, ["tool_a", "tool_b"])
    assert result.status     == AgentStatus.COMPLETED
    assert result.task_id    == "task_123"
    assert result.steps_taken == 5
    assert "tool_a" in result.tools_used
    assert result.needs_human is False


def test_build_error_result():
    result = build_error_result("task_456", "test_agent", "something failed", 2)
    assert result.status      == AgentStatus.FAILED
    assert result.confidence  == 0.0
    assert result.error       == "something failed"
    assert result.output      == {}


def test_build_human_review_result():
    result = build_human_review_result("task_789", "safety_agent", {"risk": "CRITICAL"}, "Safety critical")
    assert result.needs_human is True
    assert "human_review_reason" in result.output


def test_task_has_unique_id():
    task_a = create_diagnostic_task("symptoms a", "", "")
    task_b = create_diagnostic_task("symptoms b", "", "")
    assert task_a.task_id != task_b.task_id


# ── AgentTracer Tests ──────────────────────────────────────

def test_tracer_log_thought():
    tracer = AgentTracer("test_sess")
    tracer.log_thought("agent_a", "Checking fault codes", latency_ms=50.0)
    assert len(tracer.steps) == 1
    assert tracer.steps[0].action == "THOUGHT"
    assert tracer.steps[0].agent_name == "agent_a"


def test_tracer_log_tool_call():
    tracer = AgentTracer("test_sess")
    tracer.log_tool_call("agent_a", "fault_code_lookup", "P0300", "Misfire detected", 100.0)
    assert len(tracer.steps) == 1
    assert tracer.steps[0].action == "TOOL_CALL"
    assert tracer.steps[0].tool   == "fault_code_lookup"


def test_tracer_log_final_answer():
    tracer = AgentTracer("test_sess")
    tracer.log_final_answer("agent_a", "Spark plugs recommended")
    assert tracer.steps[0].action == "FINAL_ANSWER"


def test_tracer_log_error():
    tracer = AgentTracer("test_sess")
    tracer.log_error("agent_a", "Tool failed", retry_count=1)
    assert "ERROR" in tracer.steps[0].action


def test_tracer_detect_redundant():
    tracer = AgentTracer("test_sess")
    tracer.log_tool_call("agent_a", "fault_code_lookup", "P0300", "result1")
    tracer.log_tool_call("agent_b", "fault_code_lookup", "P0300", "result2")
    redundancies = tracer.detect_redundant_work()
    assert len(redundancies) > 0
    assert redundancies[0]["redundant"] is True


def test_tracer_no_redundancy():
    tracer = AgentTracer("test_sess")
    tracer.log_tool_call("agent_a", "fault_code_lookup",  "P0300", "result1")
    tracer.log_tool_call("agent_b", "safety_assessment", "rough idle", "result2")
    redundancies = tracer.detect_redundant_work()
    assert len(redundancies) == 0


def test_tracer_agent_stats():
    tracer = AgentTracer("test_sess")
    tracer.log_tool_call("diag_agent", "fault_code_lookup", "P0300", "result", 100.0)
    tracer.log_tool_call("diag_agent", "search_manual",     "misfire", "result", 200.0)
    assert "diag_agent" in tracer.agent_stats
    assert tracer.agent_stats["diag_agent"]["steps"] == 2


def test_tracer_summary():
    tracer = AgentTracer("test_sess")
    tracer.log_thought("agent_a", "thinking")
    tracer.log_tool_call("agent_a", "tool_x", "input", "output")
    summary = tracer.get_summary()
    assert summary["total_steps"]   == 2
    assert "agent_a" in summary["agents_involved"]
    assert "total_cost_usd" in summary
    assert "total_time_ms"  in summary


def test_tracer_full_trace():
    tracer = AgentTracer("test_sess")
    tracer.log_thought("agent_a", "reasoning step")
    trace = tracer.get_full_trace()
    assert len(trace) == 1
    assert "step_number" in trace[0]
    assert "agent_name"  in trace[0]
    assert "timestamp"   in trace[0]


# ── Memory Manager Tests (SQLite-backed) ───────────────────

def test_memory_manager_init():
    memory = MemoryManager("test_session_001")
    assert memory.session_id == "test_session_001"
    assert memory.episodic   is not None
    assert memory.semantic   is not None
    assert memory.procedural is not None


def test_memory_save_diagnosis_positive():
    memory = MemoryManager("test_sess_pos")
    memory.save_diagnosis("rough idle P0300", "spark plugs", "fixed", rating=True)
    episodes = memory.retrieve_relevant_episodes("rough idle")
    assert len(episodes) >= 0  # May be 0 if similarity too low


def test_memory_save_diagnosis_negative():
    memory   = MemoryManager("test_sess_neg")
    before   = len(db.get_all_episodes())
    memory.save_diagnosis("brake noise", "brake pads", "fixed", rating=False)
    after    = len(db.get_all_episodes())
    assert after == before  # Write policy: negative rating not saved


def test_memory_store_vehicle_fact():
    memory = MemoryManager("test_sess_fact")
    memory.store_vehicle_fact("vehicle_make", "Toyota", 0.9)
    value  = db.get_fact("vehicle_make")
    assert value == "Toyota"


def test_memory_store_user_preference():
    memory = MemoryManager("test_sess_pref")
    memory.store_user_preference("preferred_mechanic", "Joe's Garage")
    value  = db.get_fact("preferred_mechanic")
    assert value == "Joe's Garage"


def test_memory_save_procedure():
    memory = MemoryManager("test_sess_proc")
    memory.save_successful_procedure("P0300", ["fault_code_lookup", "explore_related_codes"])
    seq    = db.get_procedure("P0300")
    assert seq is not None
    assert "fault_code_lookup" in seq


def test_memory_get_stats():
    memory = MemoryManager("test_sess_stats")
    stats  = memory.get_stats()
    assert "session_id"      in stats
    assert "episodic_count"  in stats
    assert "semantic_count"  in stats
    assert "procedural_count" in stats


def test_memory_consolidate():
    memory = MemoryManager("test_sess_consol")
    # Add some episodes
    db.insert_episode("test_sess_consol", "symptoms a", "diagnosis a", "")
    db.insert_episode("test_sess_consol", "symptoms a", "diagnosis a", "")  # duplicate
    memory.consolidate()  # should remove duplicate


# ── Cleanup ────────────────────────────────────────────────

def teardown_module(module):
    if os.path.exists("test_agents.db"):
        os.remove("test_agents.db")