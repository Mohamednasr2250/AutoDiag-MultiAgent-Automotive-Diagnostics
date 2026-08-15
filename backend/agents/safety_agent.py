"""
agents/safety_agent.py — ReAct + Safety Assessment
Constructed as SafetyAgent(blackboard, tracer, tools).
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from base_agent import BaseAgent
from agent_protocol import build_success_result, build_error_result, build_human_review_result
from tools import safety_assessment

SAFETY_PROMPT = """You are the Safety Agent for AutoDiag Pro.
You have access to these tools:
{tools}

Use this format:
Question: the input question you must answer
Thought: think about safety implications
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: result of the tool
... (repeat as needed)
Thought: I now have enough information
Final Answer: risk level (LOW/MEDIUM/HIGH/CRITICAL) and explanation

Rules:
- Always run safety_assessment on the reported symptoms
- Cross-check fault codes for safety-critical systems (brakes, airbags, steering)
- Be conservative — when uncertain, escalate severity

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


class SafetyAgent(BaseAgent):
    def __init__(self, blackboard, tracer, tools):
        super().__init__("safety_agent", blackboard, tracer, tools)

    def run(self, task):
        payload = task.payload
        query = (f"Symptoms: {payload.get('symptoms')}\n"
                 f"Fault Codes: {payload.get('fault_codes') or 'None'}\n"
                 f"Assess whether this vehicle is safe to drive.")

        prompt = PromptTemplate(template=SAFETY_PROMPT,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"])

        try:
            agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            executor = AgentExecutor(agent=agent, tools=self.tools, max_iterations=self.MAX_ITERATIONS,
                handle_parsing_errors=True, return_intermediate_steps=True, verbose=False)
            result = executor.invoke({"input": query})

            direct_check = safety_assessment(payload.get("symptoms", ""))
            risk_level = self._parse_risk(direct_check)

            tools_used = [a.tool for a, _ in result.get("intermediate_steps", [])]
            for action, observation in result.get("intermediate_steps", []):
                self.tracer.log_tool_call(self.name, action.tool, str(action.tool_input)[:150], str(observation)[:200])

            output = {"risk_level": risk_level, "assessment": direct_check,
                      "agent_reasoning": result.get("output", "")[:300]}
            self.tracer.log_final_answer(self.name, f"Risk: {risk_level}")

            if risk_level == "CRITICAL":
                return build_human_review_result(task.task_id, self.name, output,
                    f"CRITICAL safety risk detected: {direct_check[:150]}")

            return build_success_result(task.task_id, self.name, output,
                steps_taken=len(tools_used), tools_used=list(set(tools_used)), confidence=0.85)
        except Exception as e:
            self.tracer.log_error(self.name, str(e))
            return build_error_result(task.task_id, self.name, str(e))

    def _parse_risk(self, assessment_text):
        if "CRITICAL" in assessment_text:
            return "CRITICAL"
        if "HIGH" in assessment_text:
            return "HIGH"
        if "MEDIUM" in assessment_text:
            return "MEDIUM"
        return "LOW"