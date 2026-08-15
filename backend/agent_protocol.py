import uuid
from schemas import AgentHandoff, AgentResult, TaskType, AgentStatus

def create_task(task_type, payload, from_agent, to_agent, context_summary="", priority=1):
    return AgentHandoff(task_id=str(uuid.uuid4()),task_type=task_type,payload=payload,
        from_agent=from_agent,to_agent=to_agent,priority=priority,context_summary=context_summary)

def create_diagnostic_task(symptoms, fault_codes, vehicle_info, context_summary=""):
    return create_task(TaskType.DIAGNOSE,{"symptoms":symptoms,"fault_codes":fault_codes,"vehicle_info":vehicle_info},"orchestrator","diagnostic_agent",context_summary,1)

def create_safety_task(symptoms, fault_codes, diagnostic_result=None, context_summary=""):
    return create_task(TaskType.SAFETY,{"symptoms":symptoms,"fault_codes":fault_codes,"diagnostic_result":diagnostic_result or {}},"orchestrator","safety_agent",context_summary,2)

def create_cost_task(likely_causes, diagnostic_result=None, context_summary=""):
    return create_task(TaskType.COST,{"likely_causes":likely_causes,"diagnostic_result":diagnostic_result or {}},"orchestrator","cost_agent",context_summary,3)

def create_reflection_task(combined_results, original_query, context_summary=""):
    return create_task(TaskType.REFLECT,{"combined_results":combined_results,"original_query":original_query},"orchestrator","reflection_agent",context_summary,4)

def build_success_result(task_id, agent_name, output, steps_taken, tools_used, confidence=0.8):
    return AgentResult(task_id=task_id,agent_name=agent_name,status=AgentStatus.COMPLETED,
        output=output,confidence=confidence,steps_taken=steps_taken,tools_used=tools_used,needs_human=False)

def build_error_result(task_id, agent_name, error, steps_taken=0):
    return AgentResult(task_id=task_id,agent_name=agent_name,status=AgentStatus.FAILED,
        output={},confidence=0.0,steps_taken=steps_taken,tools_used=[],error=error,needs_human=False)

def build_human_review_result(task_id, agent_name, output, reason):
    result = build_success_result(task_id,agent_name,output,0,[],0.9)
    result.needs_human = True
    result.output["human_review_reason"] = reason
    return result