import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import mlflow
from db import (insert_episode, get_similar_episodes, get_all_episodes,
    delete_duplicate_episodes, decay_old_episodes, upsert_fact, get_fact,
    get_all_facts, upsert_procedure, get_procedure, get_best_procedure,
    log_consolidation, get_db_stats)

class ShortTermMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns; self._turns = []
    def add(self, role, content):
        self._turns.append({"role":role,"content":content[:500],"timestamp":datetime.now().isoformat()})
        if len(self._turns) > self.max_turns: self._turns = self._turns[-self.max_turns:]
    def get_recent(self, k=3): return self._turns[-k:]
    def get_history_text(self):
        return "\n".join([f"{t['role']}: {t['content'][:150]}" for t in self._turns[-6:]])
    def clear(self): self._turns.clear()

class EpisodicMemory:
    def add_episode(self, session_id, symptoms, diagnosis, outcome=""):
        insert_episode(session_id, symptoms, diagnosis, outcome)
    def retrieve_similar(self, symptoms, k=3): return get_similar_episodes(symptoms, k)
    def get_all(self): return get_all_episodes()

class SemanticMemory:
    def store_fact(self, key, value, confidence=0.8, source="agent"):
        upsert_fact(key, value, confidence, source)
    def retrieve_fact(self, key): return get_fact(key)
    def get_all_facts(self): return get_all_facts()

class ProceduralMemory:
    def save_procedure(self, trigger, tool_sequence, success_rate=1.0):
        upsert_procedure(trigger, tool_sequence, success_rate)
    def get_procedure(self, trigger): return get_procedure(trigger)
    def get_best_procedure(self, fault_codes): return get_best_procedure(fault_codes)

class MemoryManager:
    CONSOLIDATION_N = 10
    def __init__(self, session_id=""):
        self.session_id   = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.short_term   = ShortTermMemory()
        self.episodic     = EpisodicMemory()
        self.semantic     = SemanticMemory()
        self.procedural   = ProceduralMemory()
        self._episode_count = 0

    def add_turn(self, role, content): self.short_term.add(role, content)

    def save_diagnosis(self, symptoms, diagnosis, outcome, rating=True):
        if not rating: return
        self.episodic.add_episode(self.session_id, symptoms, diagnosis, outcome)
        self._episode_count += 1
        if self._episode_count % self.CONSOLIDATION_N == 0: self.consolidate()

    def store_vehicle_fact(self, key, value, confidence=0.8):
        self.semantic.store_fact(key, value, confidence, source="agent")

    def store_user_preference(self, key, value):
        self.semantic.store_fact(key, value, confidence=1.0, source="user")

    def save_successful_procedure(self, fault_codes, tool_sequence):
        self.procedural.save_procedure(fault_codes, tool_sequence, success_rate=1.0)

    def retrieve_relevant_episodes(self, symptoms): return self.episodic.retrieve_similar(symptoms, k=3)

    def get_best_procedure(self, fault_codes): return self.procedural.get_best_procedure(fault_codes)

    def build_memory_context(self, symptoms):
        parts = []
        history = self.short_term.get_history_text()
        if history: parts.append(f"Recent conversation:\n{history}")
        episodes = self.retrieve_relevant_episodes(symptoms)
        if episodes:
            ep_text = "\n".join([f"- Past: {ep['symptoms'][:80]} → {ep['diagnosis'][:80]}" for ep in episodes[:2]])
            parts.append(f"Similar past diagnoses:\n{ep_text}")
        facts = self.semantic.get_all_facts()
        vehicle_facts = {k:v for k,v in facts.items() if any(w in k for w in ["vehicle","make","model","year","mileage"])}
        if vehicle_facts:
            parts.append("Known vehicle facts:\n" + "\n".join([f"- {k}: {v}" for k,v in list(vehicle_facts.items())[:5]]))
        return "\n\n".join(parts) if parts else ""

    def consolidate(self):
        episodes_before = len(self.episodic.get_all())
        removed_dups    = delete_duplicate_episodes()
        removed_decay   = decay_old_episodes(days=30)
        total_removed   = removed_dups + removed_decay
        episodes_after  = len(self.episodic.get_all())
        log_consolidation(episodes_before, episodes_after, total_removed)
        try:
            with mlflow.start_run(run_name="memory_consolidation"):
                mlflow.log_metric("episodes_before", episodes_before)
                mlflow.log_metric("episodes_after",  episodes_after)
                mlflow.log_metric("removed",         total_removed)
        except Exception: pass

    def get_stats(self):
        db = get_db_stats()
        return {"session_id":self.session_id,"episodic_count":db["episodic_count"],
                "semantic_count":db["semantic_count"],"procedural_count":db["procedural_count"],
                "has_short_term":len(self.short_term._turns)>0}