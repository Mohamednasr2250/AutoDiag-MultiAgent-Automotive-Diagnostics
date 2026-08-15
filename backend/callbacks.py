import time, os
from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler

class TracerCallback(BaseCallbackHandler):
    def __init__(self, tracer=None, agent_name="unknown"):
        self.tracer = tracer; self.agent_name = agent_name
        self._call_start_times = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._call_start_times[str(kwargs.get("run_id",""))] = time.time()

    def on_llm_end(self, response, **kwargs):
        run_id = str(kwargs.get("run_id",""))
        start = self._call_start_times.pop(run_id, time.time())
        latency_ms = (time.time()-start)*1000
        if self.tracer:
            output = response.generations[0][0].text[:200] if response.generations else ""
            self.tracer.log_step(self.agent_name,"LLM_CALL",observation=output,latency_ms=latency_ms)

    def on_llm_error(self, error, **kwargs):
        if self.tracer: self.tracer.log_error(self.agent_name,str(error))

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._call_start_times[str(kwargs.get("run_id",""))] = time.time()
        if self.tracer: self.tracer.log_step(self.agent_name,"TOOL_START",tool=serialized.get("name",""),tool_input=input_str)

    def on_tool_end(self, output, **kwargs):
        run_id = str(kwargs.get("run_id",""))
        start = self._call_start_times.pop(run_id, time.time())
        if self.tracer: self.tracer.log_tool_call(self.agent_name,kwargs.get("name","tool"),"",output[:300],(time.time()-start)*1000)

    def on_tool_error(self, error, **kwargs):
        if self.tracer: self.tracer.log_error(self.agent_name,f"Tool error: {str(error)}")

    def on_agent_action(self, action, **kwargs):
        if self.tracer: self.tracer.log_thought(self.agent_name,str(action.log)[:200])

    def on_agent_finish(self, finish, **kwargs):
        if self.tracer: self.tracer.log_final_answer(self.agent_name,str(finish.return_values.get("output",""))[:200])

class CostTrackingCallback(BaseCallbackHandler):
    COST_PER_CALL = 0.0001
    def __init__(self):
        self.total_calls = 0; self.total_cost = 0.0; self.call_times = []; self._call_start = None
    def on_llm_start(self, serialized, prompts, **kwargs): self._call_start = time.time()
    def on_llm_end(self, response, **kwargs):
        if self._call_start: self.call_times.append(time.time()-self._call_start); self._call_start = None
        self.total_calls += 1; self.total_cost += self.COST_PER_CALL
    def get_report(self):
        return {"total_llm_calls":self.total_calls,"total_cost_usd":round(self.total_cost,6),
                "avg_latency_ms":round((sum(self.call_times)/len(self.call_times)*1000) if self.call_times else 0,2)}

class AutoMLflowCallback(BaseCallbackHandler):
    def __init__(self, run_name="agent_run"):
        self.run_name = run_name; self._llm_calls = 0; self._tool_calls = 0; self._start = time.time()
    def on_llm_start(self, serialized, prompts, **kwargs): self._llm_calls += 1
    def on_tool_end(self, output, **kwargs): self._tool_calls += 1
    def on_chain_end(self, outputs, **kwargs):
        try:
            import mlflow
            with mlflow.start_run(run_name=self.run_name,nested=True):
                mlflow.log_metric("llm_calls",self._llm_calls)
                mlflow.log_metric("tool_calls",self._tool_calls)
                mlflow.log_metric("total_time",round(time.time()-self._start,3))
        except Exception: pass

def get_default_callbacks(tracer=None, agent_name="agent"):
    callbacks = [CostTrackingCallback()]
    if os.environ.get("ENV","development") == "development":
        try:
            from langchain.callbacks import StdOutCallbackHandler
            callbacks.append(StdOutCallbackHandler())
        except Exception: pass
    if tracer: callbacks.append(TracerCallback(tracer, agent_name))
    return callbacks