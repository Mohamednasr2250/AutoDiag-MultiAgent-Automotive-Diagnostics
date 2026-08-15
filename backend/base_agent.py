import time, os
from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEndpoint
from schemas import AgentResult, AgentHandoff, AgentStatus

class BaseAgent(ABC):
    MAX_ITERATIONS = 6; MAX_RETRIES = 3; BACKOFF_BASE = 2.0; COST_PER_CALL = 0.0001

    def __init__(self, name, blackboard, tracer, tools=None):
        self.name = name; self.blackboard = blackboard; self.tracer = tracer
        self.tools = tools or []
        self.llm = self._init_llm(); self.call_count = 0; self.total_cost = 0.0

    def _init_llm(self):
        return HuggingFaceEndpoint(repo_id="google/flan-t5-base",
            huggingfacehub_api_token=os.environ.get("HF_API_KEY"),max_new_tokens=512,temperature=0.1)

    def invoke_llm(self, prompt):
        start = time.time()
        try:
            result = self.llm.invoke(prompt)
            self.call_count += 1; self.total_cost += self.COST_PER_CALL
            self.tracer.log_step(self.name,"LLM_CALL",latency_ms=(time.time()-start)*1000,cost_usd=self.COST_PER_CALL)
            return result
        except Exception as e:
            self.tracer.log_step(self.name,"LLM_CALL",latency_ms=(time.time()-start)*1000,cost_usd=self.COST_PER_CALL)
            raise e

    def run_with_retry(self, task):
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.blackboard.is_task_done(task.task_type):
                    self.tracer.log_step(self.name,"SKIPPED",observation=f"Task {task.task_type} already done")
                    existing = self.blackboard.read(task.task_type)
                    return AgentResult(task_id=task.task_id,agent_name=self.name,status=AgentStatus.COMPLETED,
                        output=existing.content if existing else {},confidence=existing.confidence if existing else 0.5,steps_taken=0,tools_used=[])
                return self.run(task)
            except Exception as e:
                last_error = str(e)
                self.tracer.log_error(self.name,last_error,attempt+1)
                if attempt < self.MAX_RETRIES-1: time.sleep(self.BACKOFF_BASE**attempt)
        from agent_protocol import build_error_result
        return build_error_result(task.task_id,self.name,f"Failed after {self.MAX_RETRIES} retries: {last_error}")

    def check_stopping_criteria(self, iteration, output, goal_reached=False):
        if iteration >= self.MAX_ITERATIONS:
            self.tracer.log_step(self.name,"MAX_ITERATIONS_REACHED"); return True
        if "FINAL ANSWER" in output.upper(): return True
        if goal_reached: return True
        return False

    def get_cost_report(self):
        return {"agent":self.name,"llm_calls":self.call_count,"total_cost":round(self.total_cost,6)}

    @abstractmethod
    def run(self, task: AgentHandoff) -> AgentResult: pass