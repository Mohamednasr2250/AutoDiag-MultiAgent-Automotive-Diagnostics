"""
tests/test_db.py — SQLite Database Tests
Tests all tables: feedback, episodic_memory, semantic_memory, procedural_memory
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Use test database
os.environ["DB_PATH"] = "test_autodiag.db"
import db
db.DB_PATH = "test_autodiag.db"
db.init_db()

import pytest
from db import (
    insert_feedback, get_all_feedback, get_feedback_stats,
    get_feedback_by_session, insert_episode, get_similar_episodes,
    get_all_episodes, delete_duplicate_episodes, decay_old_episodes,
    upsert_fact, get_fact, get_all_facts,
    upsert_procedure, get_procedure, get_best_procedure,
    log_consolidation, get_db_stats
)


# ── Feedback Tests ─────────────────────────────────────────

def test_insert_feedback():
    entry = insert_feedback("P0300 diagnosis", "spark plugs", True, "great", "sess_1")
    assert entry is not None
    assert entry["rating"] == 1
    assert entry["query"] == "P0300 diagnosis"


def test_get_all_feedback():
    insert_feedback("test query", "test answer", False, "", "sess_2")
    all_fb = get_all_feedback()
    assert len(all_fb) >= 1


def test_feedback_stats():
    stats = get_feedback_stats()
    assert "total" in stats
    assert "positive" in stats
    assert "negative" in stats
    assert "satisfaction_rate" in stats
    assert stats["total"] >= 1


def test_get_feedback_by_session():
    insert_feedback("session query", "session answer", True, "", "unique_session_xyz")
    results = get_feedback_by_session("unique_session_xyz")
    assert len(results) >= 1
    assert results[0]["session_id"] == "unique_session_xyz"


# ── Episodic Memory Tests ──────────────────────────────────

def test_insert_episode():
    row_id = insert_episode("sess_ep", "rough idle P0300", "spark plugs recommended", "fixed")
    assert row_id > 0


def test_get_similar_episodes():
    insert_episode("sess_sim", "check engine light rough idle", "misfire diagnosis", "ok")
    results = get_similar_episodes("rough idle check engine", k=3)
    assert isinstance(results, list)


def test_get_all_episodes():
    episodes = get_all_episodes()
    assert isinstance(episodes, list)
    assert len(episodes) >= 1


def test_delete_duplicate_episodes():
    # Insert same episode twice
    insert_episode("sess_dup", "same symptoms test", "same diagnosis test", "")
    insert_episode("sess_dup", "same symptoms test", "same diagnosis test", "")
    removed = delete_duplicate_episodes()
    assert removed >= 0


def test_decay_episodes():
    removed = decay_old_episodes(days=30)
    assert removed >= 0


# ── Semantic Memory Tests ──────────────────────────────────

def test_upsert_fact():
    result = upsert_fact("vehicle_make", "Toyota", 0.9, "user")
    assert result is True


def test_upsert_fact_low_confidence():
    result = upsert_fact("low_conf_key", "some value", 0.3, "agent")
    assert result is False


def test_get_fact():
    upsert_fact("test_key_abc", "test_value_abc", 0.8, "agent")
    value = get_fact("test_key_abc")
    assert value == "test_value_abc"


def test_get_fact_missing():
    value = get_fact("nonexistent_key_xyz")
    assert value is None


def test_get_all_facts():
    upsert_fact("fact_1", "value_1", 0.8, "agent")
    upsert_fact("fact_2", "value_2", 0.9, "user")
    facts = get_all_facts()
    assert isinstance(facts, dict)
    assert len(facts) >= 2


# ── Procedural Memory Tests ────────────────────────────────

def test_upsert_procedure():
    result = upsert_procedure("P0300", ["fault_code_lookup", "search_vehicle_manual"], 1.0)
    assert result is True


def test_upsert_procedure_low_success():
    result = upsert_procedure("P0999", ["tool_a"], 0.5)
    assert result is False


def test_get_procedure():
    upsert_procedure("P0171_test", ["fault_code_lookup", "explore_related_codes"], 1.0)
    seq = get_procedure("P0171_test")
    assert seq is not None
    assert "fault_code_lookup" in seq


def test_get_procedure_missing():
    seq = get_procedure("NONEXISTENT_TRIGGER")
    assert seq is None


def test_get_best_procedure():
    upsert_procedure("P0300,P0171", ["fault_code_lookup", "find_common_cause"], 0.9)
    seq = get_best_procedure("P0300")
    assert seq is not None


# ── Consolidation + Stats ──────────────────────────────────

def test_log_consolidation():
    log_consolidation(10, 8, 2)


def test_get_db_stats():
    stats = get_db_stats()
    assert "feedback_count"   in stats
    assert "episodic_count"   in stats
    assert "semantic_count"   in stats
    assert "procedural_count" in stats


# ── Cleanup ────────────────────────────────────────────────

def teardown_module(module):
    """Remove test database after tests."""
    if os.path.exists("test_autodiag.db"):
        os.remove("test_autodiag.db")