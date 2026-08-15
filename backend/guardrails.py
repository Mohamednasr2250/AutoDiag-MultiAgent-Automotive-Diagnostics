from typing import Dict, Any, Tuple, List
from schemas import DiagnosticRequest, AgentResult

BLOCKED_TOOL_CALLS = ["delete_database","drop_table","rm -rf","format_disk"]
HUMAN_REVIEW_TRIGGERS = ["CRITICAL","do not drive","airbag","brake failure","fire","immediate danger"]
INJECTION_PATTERNS = ["ignore previous","ignore all","system prompt","you are now","forget everything","new instructions"]

def validate_input(request: DiagnosticRequest) -> Tuple[bool, str]:
    symptoms = request.symptoms.strip()
    if len(symptoms) < 5: return False, "Symptoms too short — describe in more detail."
    if len(symptoms) > 2000: return False, "Symptoms too long — limit to 2000 characters."
    for pattern in INJECTION_PATTERNS:
        if pattern in symptoms.lower(): return False, "Invalid input detected."
    if request.fault_codes:
        for code in request.fault_codes.upper().replace(",", " ").split():
            code = code.strip()
            if code and not (len(code)==5 and code[0] in ("P","C","B","U") and code[1:].isdigit()):
                return False, f"Invalid fault code format: {code}"
    return True, "OK"

def sanitize_input(request: DiagnosticRequest) -> DiagnosticRequest:
    request.symptoms = request.symptoms.strip()[:2000]
    request.fault_codes = request.fault_codes.strip().upper() if request.fault_codes else ""
    request.vehicle_info = request.vehicle_info.strip()[:200] if request.vehicle_info else ""
    return request

def validate_tool_call(tool_name: str, tool_input: Any) -> Tuple[bool, str]:
    for blocked in BLOCKED_TOOL_CALLS:
        if blocked in str(tool_name).lower() or blocked in str(tool_input).lower():
            return False, f"Tool call blocked: '{blocked}'"
    return True, "OK"

def validate_agent_output(result: AgentResult) -> Tuple[bool, str]:
    if not result.output: return False, f"Agent {result.agent_name} returned empty output."
    if result.confidence < 0.1: return False, f"Agent {result.agent_name} has very low confidence."
    return True, "OK"

def needs_human_review(diagnosis: str, safety_result: Dict[str, Any], fault_codes: str = "") -> Tuple[bool, str]:
    diagnosis_lower = diagnosis.lower()
    for trigger in HUMAN_REVIEW_TRIGGERS:
        if trigger.lower() in diagnosis_lower: return True, f"Human review required: '{trigger}' detected"
    if safety_result.get("risk_level") in ("CRITICAL","HIGH"): return True, f"Human review required: {safety_result.get('risk_level')} risk"
    if fault_codes:
        for code in fault_codes.upper().split():
            if code.startswith(("C0","B0")): return True, f"Human review required: safety code {code}"
    return False, "No human review needed"

def build_human_review_response(reason: str, partial_result: Dict) -> Dict:
    return {"status":"PENDING_HUMAN_REVIEW","reason":reason,"partial_diagnosis":partial_result,
            "message":"⚠️ This case requires review by a certified mechanic.",
            "recommended_action":"Contact a certified automotive technician immediately."}

def get_guardrail_summary() -> Dict:
    return {"input_guardrails":["Min 5 char symptoms","Max 2000 chars","Prompt injection detection","OBD-II format validation"],
            "output_guardrails":["Blocked dangerous tool calls","Empty output detection","Low confidence flagging"],
            "human_in_loop_triggers":HUMAN_REVIEW_TRIGGERS,"blocked_tool_calls":BLOCKED_TOOL_CALLS}