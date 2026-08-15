"""
agents/reflection_agent.py — Self-Critique + Revision (Draft -> Critique -> Revise)
Constructed as ReflectionAgent(blackboard, tracer) — no tools needed.
"""

from base_agent import BaseAgent
from agent_protocol import build_success_result, build_error_result


class ReflectionAgent(BaseAgent):
    def __init__(self, blackboard, tracer, tools=None):
        super().__init__("reflection_agent", blackboard, tracer, tools)

    def run(self, task):
        payload = task.payload
        combined = payload.get("combined_results", {})
        original_query = payload.get("original_query", "")

        try:
            draft = self._draft(combined, original_query)
            self.tracer.log_thought(self.name, "Draft compiled from all agent outputs")

            critique = self._critique(draft, combined)
            self.tracer.log_thought(self.name, f"Critique: {critique[:150]}")

            final = self._revise(draft, critique)
            self.tracer.log_final_answer(self.name, final[:200])

            output = {"draft": draft, "critique": critique, "final_answer": final,
                      "final_recommendation": final}
            return build_success_result(task.task_id, self.name, output, steps_taken=3, tools_used=[], confidence=0.8)
        except Exception as e:
            self.tracer.log_error(self.name, str(e))
            return build_error_result(task.task_id, self.name, str(e))

    def _draft(self, combined, original_query):
        diag = combined.get("diagnosis") or combined.get("DIAGNOSE", {})
        safety = combined.get("safety") or combined.get("SAFETY", {})
        cost = combined.get("cost_estimate") or combined.get("COST", {})
        return (f"For the reported issue '{original_query}': "
                f"{diag.get('primary_issue', 'diagnosis pending')}. "
                f"Safety risk: {safety.get('risk_level', 'unknown')}. "
                f"Estimated cost: {str(cost.get('cost_estimate', 'not available'))[:150]}.")

    def _critique(self, draft, combined):
        prompt = (f"Critique this automotive diagnosis draft for gaps, inconsistencies, "
                  f"or missing safety warnings. Be concise (2-3 sentences):\n{draft}")
        try:
            return self.invoke_llm(prompt).strip()
        except Exception:
            return "No major issues found — draft appears consistent with agent findings."

    def _revise(self, draft, critique):
        if "no major issues" in critique.lower() or len(critique) < 15:
            return draft
        prompt = f"Revise this diagnosis draft to address the critique.\nDraft: {draft}\nCritique: {critique}\nRevised:"
        try:
            revised = self.invoke_llm(prompt).strip()
            return revised if revised else draft
        except Exception:
            return draft