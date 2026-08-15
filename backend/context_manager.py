from typing import List, Dict, Any, Optional
import mlflow

MAX_CONTEXT_TOKENS = 400; AVG_CHARS_PER_TOKEN = 4

class ContextManager:
    def __init__(self, max_tokens=MAX_CONTEXT_TOKENS, window_size=5):
        self.max_tokens = max_tokens; self.window_size = window_size
        self.max_chars = max_tokens * AVG_CHARS_PER_TOKEN

    def count_tokens(self, text): return len(text) // AVG_CHARS_PER_TOKEN

    def sliding_window(self, history, k=None):
        k = k or self.window_size
        return history[-k:] if len(history) > k else history

    def truncate_tool_output(self, output, max_chars=500):
        if len(output) <= max_chars: return output
        return output[:max_chars] + f"\n... [truncated {len(output)-max_chars} chars]"

    def filter_stale_observations(self, steps):
        seen_tools = {}; filtered = []
        for step in steps:
            tool = step.get("tool","")
            if tool: seen_tools[tool] = step.get("observation","")
            elif "FINAL" in step.get("action","").upper(): filtered.append(step)
        for tool, obs in seen_tools.items():
            filtered.append({"action":"TOOL_RESULT","tool":tool,"observation":self.truncate_tool_output(obs)})
        return filtered

    def summarize_history(self, history, llm):
        if not history: return ""
        history_text = "\n".join([f"{t.get('role','')}: {t.get('content','')[:200]}" for t in history])
        prompt = f"Summarize in 2-3 sentences:\n{history_text}\nSummary:"
        try:
            summary = llm.invoke(prompt).strip()
        except Exception:
            summary = " | ".join([f"{t.get('role','')}: {t.get('content','')[:100]}" for t in history[-2:]])
        try:
            with mlflow.start_run(run_name="context_compaction"):
                mlflow.log_metric("original_turns",len(history)); mlflow.log_metric("summary_chars",len(summary))
        except Exception: pass
        return summary

    def build_agent_context(self, task_description, relevant_history, tool_results, llm=None, agent_name=""):
        history = self.sliding_window(relevant_history)
        filtered_tools = self.filter_stale_observations(tool_results)
        context_parts = [f"Task: {task_description}"]
        if history:
            context_parts.append("Recent history:\n" + "\n".join([f"{t.get('role','')}: {t.get('content','')[:150]}" for t in history]))
        if filtered_tools:
            context_parts.append("Previous results:\n" + "\n".join([f"{t.get('tool','')}: {self.truncate_tool_output(t.get('observation',''),300)}" for t in filtered_tools]))
        full_context = "\n\n".join(context_parts)
        if self.count_tokens(full_context) > self.max_tokens and llm:
            summary = self.summarize_history(history, llm)
            full_context = f"Task: {task_description}\nSummary: {summary}"
        return full_context

    def prepare_isolated_context(self, task_type, payload, context_summary=""):
        parts = [f"Your task: {task_type}"]
        if context_summary: parts.append(f"Background: {context_summary[:300]}")
        for key, val in payload.items():
            if val: parts.append(f"{key}: {str(val)[:200]}")
        return "\n".join(parts)