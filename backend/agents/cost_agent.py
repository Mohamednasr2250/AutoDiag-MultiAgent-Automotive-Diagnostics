"""
agents/cost_agent.py — ReAct + Repair Cost Estimation + Dijkstra Path Costing
Constructed as CostAgent(blackboard, tracer, tools).
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from base_agent import BaseAgent
from agent_protocol import build_success_result, build_error_result
from tools import repair_estimate
from vehicle_hierarchy import get_repair_order
from fault_graph import find_cheapest_path

COST_PROMPT = """You are the Cost Agent for AutoDiag Pro.
You have access to these tools:
{tools}

Use this format:
Question: the input question you must answer
Thought: think about what needs pricing
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: result of the tool
... (repeat as needed)
Thought: I now have enough information
Final Answer: total cost estimate and repair order

Rules:
- Estimate cost for every likely cause identified by the diagnostic agent
- Use get_repair_order to sequence repairs correctly — never suggest fixing an effect before its cause
- If two or more related fault codes were reported, use find_cheapest_path to identify the
  cheapest diagnostic route between them instead of pricing each in isolation

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


class CostAgent(BaseAgent):
    def __init__(self, blackboard, tracer, tools):
        super().__init__("cost_agent", blackboard, tracer, tools)

    def run(self, task):
        payload = task.payload
        causes = payload.get("likely_causes", [])
        causes_str = ", ".join(causes) if causes else "unknown components"
        fault_codes = (payload.get("diagnostic_result") or {}).get("fault_codes", "")

        query = f"Likely causes: {causes_str}\nEstimate repair costs and correct repair order."

        prompt = PromptTemplate(template=COST_PROMPT,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"])

        try:
            agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
            executor = AgentExecutor(agent=agent, tools=self.tools, max_iterations=self.MAX_ITERATIONS,
                handle_parsing_errors=True, return_intermediate_steps=True, verbose=False)
            result = executor.invoke({"input": query})

            direct_estimate = repair_estimate(causes_str)
            repair_order = get_repair_order(fault_codes) if fault_codes else "No fault codes provided for ordering."

            tools_used = [a.tool for a, _ in result.get("intermediate_steps", [])]
            for action, observation in result.get("intermediate_steps", []):
                self.tracer.log_tool_call(self.name, action.tool, str(action.tool_input)[:150], str(observation)[:200])

            output = {"cost_estimate": direct_estimate, "repair_order": repair_order,
                      "agent_reasoning": result.get("output", "")[:300]}

            codes_list = [c.strip() for c in fault_codes.replace(",", " ").split() if c.strip()]
            if len(codes_list) >= 2:
                cheapest = find_cheapest_path(", ".join(codes_list[:2]))
                output["cheapest_repair_path"] = cheapest
                self.tracer.log_tool_call(self.name, "find_cheapest_path", fault_codes, cheapest[:200])

            self.tracer.log_final_answer(self.name, direct_estimate[:150])

            return build_success_result(task.task_id, self.name, output,
                steps_taken=len(tools_used), tools_used=list(set(tools_used)), confidence=0.75)
        except Exception as e:
            self.tracer.log_error(self.name, str(e))
            return build_error_result(task.task_id, self.name, str(e))