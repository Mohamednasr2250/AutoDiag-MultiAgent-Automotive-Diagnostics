import time, mlflow, json, tempfile, os
from datetime import datetime
from typing import List, Dict, Optional
from schemas import TraceStep

class AgentTracer:
    COST_PER_CALL_USD = 0.0001
    def __init__(self, session_id=""):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.steps: List[TraceStep] = []
        self.start_time = time.time()
        self.total_cost = 0.0
        self.agent_stats: Dict[str, Dict] = {}

    def log_step(self, agent_name, action, tool=None, tool_input=None, observation=None, thought=None, latency_ms=0.0, cost_usd=None):
        if cost_usd is None: cost_usd = self.COST_PER_CALL_USD
        self.total_cost += cost_usd
        step = TraceStep(step_number=len(self.steps)+1,agent_name=agent_name,action=action,
            tool=tool,tool_input=str(tool_input)[:200] if tool_input else None,
            observation=str(observation)[:300] if observation else None,
            thought=str(thought)[:200] if thought else None,
            latency_ms=round(latency_ms,2),cost_usd=round(cost_usd,6),timestamp=datetime.now().isoformat())
        self.steps.append(step)
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {"steps":0,"tools_used":[],"total_ms":0.0,"cost_usd":0.0}
        s = self.agent_stats[agent_name]
        s["steps"] += 1; s["total_ms"] += latency_ms; s["cost_usd"] += cost_usd
        if tool: s["tools_used"].append(tool)

    def log_thought(self, agent_name, thought, latency_ms=0.0):
        self.log_step(agent_name,"THOUGHT",thought=thought,latency_ms=latency_ms)

    def log_tool_call(self, agent_name, tool, tool_input, observation, latency_ms=0.0):
        self.log_step(agent_name,"TOOL_CALL",tool=tool,tool_input=tool_input,observation=observation,latency_ms=latency_ms)

    def log_final_answer(self, agent_name, answer, latency_ms=0.0):
        self.log_step(agent_name,"FINAL_ANSWER",observation=answer,latency_ms=latency_ms)

    def log_error(self, agent_name, error, retry_count=0):
        self.log_step(agent_name,f"ERROR_RETRY_{retry_count}",observation=error)

    def detect_redundant_work(self):
        tool_calls = {}
        for step in self.steps:
            if step.tool and step.tool_input:
                key = f"{step.tool}:{step.tool_input[:50]}"
                tool_calls.setdefault(key,[]).append(step.agent_name)
        return [{"tool_call":k,"called_by":v,"redundant":True} for k,v in tool_calls.items() if len(v)>1]

    def get_summary(self):
        return {"session_id":self.session_id,"total_steps":len(self.steps),
            "total_time_ms":round((time.time()-self.start_time)*1000,2),
            "total_cost_usd":round(self.total_cost,6),"agents_involved":list(self.agent_stats.keys()),
            "agent_stats":self.agent_stats,"redundant_calls":self.detect_redundant_work(),
            "has_redundancy":len(self.detect_redundant_work())>0}

    def get_full_trace(self):
        return [step.dict() for step in self.steps]

    def flush_to_mlflow(self):
        summary = self.get_summary()
        try:
            with mlflow.start_run(run_name=f"trace_{self.session_id}"):
                mlflow.log_metric("total_steps",summary["total_steps"])
                mlflow.log_metric("total_time_ms",summary["total_time_ms"])
                mlflow.log_metric("total_cost_usd",summary["total_cost_usd"])
                mlflow.log_param("session_id",self.session_id)
                with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
                    json.dump(self.get_full_trace(),f,indent=2); tmp=f.name
                mlflow.log_artifact(tmp,"agent_traces"); os.unlink(tmp)
        except Exception:
            pass