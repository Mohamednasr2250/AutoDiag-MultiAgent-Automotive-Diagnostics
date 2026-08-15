"""
agents/diagnostic_agent.py — ReAct + Agentic RAG
Constructed as DiagnosticAgent(blackboard, tracer, tools).
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from base_agent import BaseAgent
from agent_protocol import build_success_result, build_error_result

DIAGNOSTIC_PROMPT = """You are the Diagnostic Agent for AutoDiag Pro, an expert automotive diagnostician.
You have access to these tools:
{tools}

Use this format:
Question: the input question you must answer
Thought: think about what to check next
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: result of the tool
... (repeat as needed)
Thought: I now have enough information
Final Answer: primary issue, likely causes, and confidence

Rules:
- Always check fault codes first if provided
- Use explore_related_codes or find_root_cause_chain when multiple codes are involved
- Search the manual for anything not covered by fault codes
- Be specific about likely causes

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


class DiagnosticAgent(BaseAgent):
    def __init__(self, blackboard, tracer, tools):
        super().__init__("diagnostic_agent", blackboard, tracer, tools)

    def run(self, task):
        payload = task.payload
        query = (f"Vehicle: {payload.get('vehicle_info') or 'Unknown'}\n"
                 f"Symptoms: {payload.get('symptoms')}\n"
                 f"Fault Codes: {payload.get('fault_codes') or 'None reported'}\n"
                 f"{('Context: ' + task.context_summary) if task.context_summary else ''}\n"
                 f"Diagnose this vehicle issue step by step.")

        prompt = PromptTemplate(template=DIAGNOSTIC_PROMPT,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"])

        try:
            agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            executor = AgentExecutor(agent=agent, tools=self.tools, max_iterations=self.MAX_ITERATIONS,
                handle_parsing_errors=True, return_intermediate_steps=True, verbose=False)
            result = executor.invoke({"input": query})

            tools_used = []
            for action, observation in result.get("intermediate_steps", []):
                tools_used.append(action.tool)
                self.tracer.log_tool_call(self.name, action.tool, str(action.tool_input)[:150], str(observation)[:200])

            output_text = result.get("output", "")
            output = {
                "primary_issue": output_text[:300],
                "likely_causes": self._extract_causes(output_text, payload.get("fault_codes", "")),
                "full_analysis": output_text,
                "fault_codes": payload.get("fault_codes", "")
            }
            self.tracer.log_final_answer(self.name, output_text[:200])

            return build_success_result(task.task_id, self.name, output,
                steps_taken=len(tools_used), tools_used=list(set(tools_used)), confidence=0.8)
        except Exception as e:
            self.tracer.log_error(self.name, str(e))
            return build_error_result(task.task_id, self.name, str(e))

    def _extract_causes(self, text, fault_codes):
        from tools import FAULT_CODES
        causes = []
        for code in fault_codes.upper().replace(",", " ").split():
            if code in FAULT_CODES:
                causes.extend(FAULT_CODES[code]["causes"])
        return list(set(causes)) if causes else ["See full analysis"]