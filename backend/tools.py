"""
tools.py — All Agent Tools
6 original tools + graph-powered tools (BFS/DFS/Dijkstra/causal/priority) +
vehicle hierarchy tools + online search + human-in-loop tool.
StructuredTool with typed Pydantic input schemas.
"""

import subprocess
import tempfile
import os
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from fault_graph import (
    explore_related_codes, find_root_cause_chain, find_common_cause,
    predict_upcoming_codes, get_repair_priority_tool, find_cheapest_path
)
from vehicle_hierarchy import get_system_context, get_repair_order, get_affected_components

FAULT_CODES = {
    "P0101": {"name": "MAF Sensor Circuit Range/Performance", "severity": "MEDIUM",
        "description": "Mass airflow sensor reading out of expected range.",
        "symptoms": ["rough idle", "stalling", "poor fuel economy"],
        "causes": ["dirty MAF sensor", "vacuum leak", "wiring fault"],
        "urgency": "Schedule service within 2 weeks", "suggested_fix": "Clean or replace MAF sensor"},
    "P0171": {"name": "System Too Lean (Bank 1)", "severity": "MEDIUM",
        "description": "Engine running lean — too much air or too little fuel.",
        "symptoms": ["poor idle", "hesitation", "check engine light"],
        "causes": ["MAF sensor", "O2 sensor", "vacuum leak", "fuel pressure"],
        "urgency": "Schedule service within 2 weeks", "suggested_fix": "Inspect vacuum lines and MAF sensor"},
    "P0300": {"name": "Random/Multiple Cylinder Misfire", "severity": "HIGH",
        "description": "Multiple cylinders misfiring — can damage the catalytic converter.",
        "symptoms": ["rough idle", "check engine light", "poor acceleration"],
        "causes": ["spark plugs", "ignition coils", "fuel injectors", "vacuum leak"],
        "urgency": "Address within days — risk of catalytic converter damage",
        "suggested_fix": "Replace spark plugs and inspect ignition coils"},
    "P0301": {"name": "Cylinder 1 Misfire", "severity": "HIGH",
        "description": "Misfire detected specifically on cylinder 1.",
        "symptoms": ["rough idle", "check engine light"],
        "causes": ["spark plug", "ignition coil", "fuel injector"],
        "urgency": "Address within days", "suggested_fix": "Inspect cylinder 1 spark plug and coil"},
    "P0302": {"name": "Cylinder 2 Misfire", "severity": "HIGH",
        "description": "Misfire detected specifically on cylinder 2.",
        "symptoms": ["rough idle", "check engine light"],
        "causes": ["spark plug", "ignition coil", "fuel injector"],
        "urgency": "Address within days", "suggested_fix": "Inspect cylinder 2 spark plug and coil"},
    "P0303": {"name": "Cylinder 3 Misfire", "severity": "HIGH",
        "description": "Misfire detected specifically on cylinder 3.",
        "symptoms": ["rough idle", "check engine light"],
        "causes": ["spark plug", "ignition coil", "fuel injector"],
        "urgency": "Address within days", "suggested_fix": "Inspect cylinder 3 spark plug and coil"},
    "P0304": {"name": "Cylinder 4 Misfire", "severity": "HIGH",
        "description": "Misfire detected specifically on cylinder 4.",
        "symptoms": ["rough idle", "check engine light"],
        "causes": ["spark plug", "ignition coil", "fuel injector"],
        "urgency": "Address within days", "suggested_fix": "Inspect cylinder 4 spark plug and coil"},
    "P0420": {"name": "Catalyst System Efficiency Below Threshold", "severity": "MEDIUM",
        "description": "Catalytic converter operating below efficiency threshold.",
        "symptoms": ["check engine light", "failed emissions test"],
        "causes": ["faulty catalytic converter", "O2 sensors", "exhaust leaks"],
        "urgency": "Schedule service within 30 days",
        "suggested_fix": "Check O2 sensors first, replace catalytic converter if needed"},
    "P0430": {"name": "Catalyst System Efficiency (Bank 2)", "severity": "MEDIUM",
        "description": "Catalytic converter Bank 2 operating below efficiency threshold.",
        "symptoms": ["check engine light", "failed emissions test"],
        "causes": ["faulty catalytic converter", "O2 sensors"],
        "urgency": "Schedule service within 30 days",
        "suggested_fix": "Check O2 sensors first, replace catalytic converter if needed"},
    "P0442": {"name": "EVAP System Small Leak", "severity": "LOW",
        "description": "Small evaporative emissions leak detected.",
        "symptoms": ["check engine light", "fuel smell"],
        "causes": ["loose gas cap", "EVAP hose crack"],
        "urgency": "Low priority — check gas cap first", "suggested_fix": "Tighten or replace gas cap"},
    "P0455": {"name": "EVAP System Large Leak", "severity": "LOW",
        "description": "Large evaporative emissions leak detected.",
        "symptoms": ["check engine light", "strong fuel smell"],
        "causes": ["EVAP hose disconnected", "purge valve stuck open"],
        "urgency": "Schedule service within 30 days", "suggested_fix": "Inspect EVAP hoses and purge valve"},
    "P0700": {"name": "Transmission Control System Malfunction", "severity": "HIGH",
        "description": "Transmission control module detected a fault.",
        "symptoms": ["transmission slipping", "harsh shifting", "won't shift"],
        "causes": ["low transmission fluid", "solenoids", "TCM failure"],
        "urgency": "Address promptly — risk of transmission damage",
        "suggested_fix": "Check transmission fluid level and condition first"},
    "P0730": {"name": "Incorrect Gear Ratio", "severity": "HIGH",
        "description": "Transmission gear ratio does not match expected value.",
        "symptoms": ["harsh shifting", "slipping"],
        "causes": ["solenoids", "worn clutch packs"],
        "urgency": "Address promptly", "suggested_fix": "Diagnose transmission solenoids"},
    "C0035": {"name": "Left Front Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "description": "ABS wheel speed sensor fault — safety critical.",
        "symptoms": ["ABS light", "traction control off", "brake issues"],
        "causes": ["wheel speed sensor", "wiring", "ABS module"],
        "urgency": "Do not drive — immediate professional inspection required",
        "suggested_fix": "Replace left front wheel speed sensor"},
    "C0040": {"name": "Right Front Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "description": "ABS wheel speed sensor fault — safety critical.",
        "symptoms": ["ABS light", "traction control off"],
        "causes": ["wheel speed sensor", "wiring"],
        "urgency": "Do not drive — immediate professional inspection required",
        "suggested_fix": "Replace right front wheel speed sensor"},
    "C0045": {"name": "Rear Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "description": "ABS wheel speed sensor fault — safety critical.",
        "symptoms": ["ABS light", "traction control off"],
        "causes": ["wheel speed sensor", "wiring"],
        "urgency": "Do not drive — immediate professional inspection required",
        "suggested_fix": "Replace rear wheel speed sensor"},
    "B0001": {"name": "Airbag Deployment Loop Fault", "severity": "CRITICAL",
        "description": "Airbag system fault — safety critical, do not drive.",
        "symptoms": ["airbag light on"],
        "causes": ["airbag sensor", "clock spring", "SRS module"],
        "urgency": "Do not drive immediately — safety system compromised",
        "suggested_fix": "Professional SRS system diagnosis required"},
    "U0100": {"name": "Lost Communication With ECM/PCM", "severity": "HIGH",
        "description": "Loss of communication with engine/powertrain control module.",
        "symptoms": ["multiple warning lights", "stalling", "no start"],
        "causes": ["wiring fault", "ECM failure", "battery issue"],
        "urgency": "Address promptly — vehicle may not start reliably",
        "suggested_fix": "Check battery and ECM wiring harness"},
}

SAFETY_TIERS = {
    "CRITICAL": ["fire", "smoke", "no brakes", "brake failure", "airbag", "air bag",
                 "steering failure", "engine fire", "smell of gas", "fuel leak"],
    "HIGH":     ["abs light", "brake pedal", "brake noise", "soft brake", "power steering",
                 "steering wheel shakes", "grinding brakes"],
    "MEDIUM":   ["traction control", "stability control", "tpms", "tire pressure warning"],
}

REPAIR_COSTS = {
    "spark plugs":         {"min": 100, "max": 300,  "time_hours": 1},
    "ignition coils":      {"min": 200, "max": 500,  "time_hours": 2},
    "fuel injectors":      {"min": 300, "max": 800,  "time_hours": 3},
    "o2 sensor":           {"min": 150, "max": 400,  "time_hours": 1},
    "maf sensor":          {"min": 100, "max": 350,  "time_hours": 1},
    "catalytic converter": {"min": 500, "max": 2000, "time_hours": 3},
    "wheel speed sensor":  {"min": 150, "max": 400,  "time_hours": 2},
    "abs module":          {"min": 800, "max": 2000, "time_hours": 4},
    "transmission fluid":  {"min": 100, "max": 250,  "time_hours": 1},
    "brake pads":          {"min": 150, "max": 400,  "time_hours": 2},
    "brake rotors":        {"min": 200, "max": 600,  "time_hours": 2},
    "gas cap":             {"min": 15,  "max": 40,   "time_hours": 0.1},
}

MAINTENANCE_SCHEDULES = {
    "toyota": "5k km: Oil + rotation | 15k km: Air filter | 30k km: Spark plugs | 50k km: Transmission fluid",
    "honda":  "8k km: Oil + rotation | 24k km: Air filter | 48k km: Spark plugs | 96k km: Timing belt",
    "bmw":    "10k km: Oil + inspection | 40k km: Brake fluid | 60k km: Spark plugs | 100k km: Timing chain check",
    "ford":   "8k km: Oil + rotation | 30k km: Air filter | 60k km: Spark plugs | 100k km: Coolant flush",
    "default":"5k km: Oil change, tire rotation | 15k km: Air + cabin filter | 30k km: Spark plugs, brake fluid | 50k km: Transmission fluid, coolant flush",
}


def fault_code_lookup(codes: str) -> str:
    codes_list = [c.strip().upper() for c in codes.replace(",", " ").split() if c.strip()]
    if not codes_list:
        return "No fault codes provided."
    results = []
    for code in codes_list:
        if code in FAULT_CODES:
            info = FAULT_CODES[code]
            results.append(
                f"Fault Code: {code}\nName: {info['name']}\nSeverity: {info['severity']}\n"
                f"Description: {info['description']}\nCommon Symptoms: {', '.join(info['symptoms'])}\n"
                f"Common Causes: {', '.join(info['causes'])}\nUrgency: {info['urgency']}\n"
                f"Suggested Fix: {info['suggested_fix']}"
            )
        else:
            results.append(f"Fault code {code} not in database. May need manufacturer-specific lookup.")
    return "\n\n".join(results)


def safety_assessment(symptoms: str) -> str:
    s = symptoms.lower()
    matched_critical = [k for k in SAFETY_TIERS["CRITICAL"] if k in s]
    if matched_critical:
        return (f"🚨 CRITICAL SAFETY ALERT 🚨\nDetected: {', '.join(matched_critical)}\n"
                f"Do NOT drive this vehicle. Pull over safely and call roadside assistance immediately.")
    matched_high = [k for k in SAFETY_TIERS["HIGH"] if k in s]
    if matched_high:
        return (f"⚠️ HIGH SEVERITY\nDetected: {', '.join(matched_high)}\n"
                f"Avoid driving until inspected by a certified mechanic — brake/steering system involved.")
    matched_medium = [k for k in SAFETY_TIERS["MEDIUM"] if k in s]
    if matched_medium:
        return (f"⚠️ MEDIUM SEVERITY\nDetected: {', '.join(matched_medium)}\n"
                f"Schedule service soon — not an immediate emergency.")
    return "✅ No immediate safety-critical symptoms detected. Monitor and schedule routine maintenance."


def repair_estimate(components: str) -> str:
    c_lower = components.lower()
    estimates = []
    total_min = total_max = 0
    total_time = 0.0
    for component, cost in REPAIR_COSTS.items():
        if component in c_lower:
            estimates.append(f"{component}: ${cost['min']}-${cost['max']} ({cost['time_hours']}h labor)")
            total_min += cost["min"]; total_max += cost["max"]; total_time += cost["time_hours"]
    if not estimates:
        return "Could not estimate cost. Please specify known components (e.g. spark plugs, O2 sensor)."
    return (f"Repair Estimates:\n" + "\n".join(estimates) +
            f"\n\nTotal Estimated Cost: ${total_min}-${total_max}\nTotal Estimated Time: {total_time}h\n"
            f"Note: Prices vary by location and vehicle make/model.")


def get_maintenance_schedule(vehicle_info: str) -> str:
    v = vehicle_info.lower()
    for make, schedule in MAINTENANCE_SCHEDULES.items():
        if make != "default" and make in v:
            return f"Maintenance Schedule ({make.title()}):\n{schedule}"
    return f"Standard Maintenance Schedule:\n{MAINTENANCE_SCHEDULES['default']}"


BLOCKED_CODE_PATTERNS = [
    "import os", "import sys", "import socket", "import subprocess", "import shutil",
    "__import__", "open(", "eval(", "exec(", "os.system", "os.popen"
]


def run_diagnostic_code(code: str) -> str:
    for pattern in BLOCKED_CODE_PATTERNS:
        if pattern in code:
            return f"Blocked: code contains disallowed pattern '{pattern}'"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(["python3", path], capture_output=True, text=True,
            timeout=5, cwd=tempfile.gettempdir())
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else "Code executed with no output."
    except subprocess.TimeoutExpired:
        return "Execution timed out (5s limit exceeded)."
    except Exception as e:
        return f"Execution error: {str(e)}"
    finally:
        os.unlink(path)


def online_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No online results found for '{query}'."
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')[:150]}" for r in results)
    except Exception as e:
        return f"Online search unavailable ({str(e)}). Rely on local fault code database and manual search."


def ask_human(question: str) -> str:
    return f"[PENDING HUMAN INPUT] Question queued for human review: {question}"


def search_vehicle_manual(vector_store):
    def _search(query: str) -> str:
        if vector_store is None:
            return "No manual search available — vector store not connected."
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in uploaded manuals."
        return "\n\n".join(f"Manual Section {i}:\n{doc.page_content[:300]}" for i, doc in enumerate(docs, 1))
    return _search


class CodeInput(BaseModel):
    codes: str = Field(description="One or more OBD-II fault codes, comma or space separated")

class SymptomsInput(BaseModel):
    symptoms: str = Field(description="Description of reported vehicle symptoms")

class ComponentsInput(BaseModel):
    components: str = Field(description="Comma-separated list of components needing repair")

class VehicleInput(BaseModel):
    vehicle_info: str = Field(description="Vehicle make/model/year string")

class QueryInput(BaseModel):
    query: str = Field(description="Search query")

class CodeSingleInput(BaseModel):
    code: str = Field(description="A single OBD-II fault code")

class CodeExecInput(BaseModel):
    code: str = Field(description="Python code to execute in sandbox")

class TwoCodeInput(BaseModel):
    codes: str = Field(description="Two OBD-II fault codes separated by a comma, e.g. 'P0300, P0420'")


def get_all_tools(vector_store) -> list:
    return get_diagnostic_tools(vector_store) + get_safety_tools() + get_cost_tools()


def get_diagnostic_tools(vector_store) -> list:
    return [
        StructuredTool.from_function(func=fault_code_lookup, name="fault_code_lookup",
            description="Look up OBD-II fault codes. Returns severity, causes, urgency, suggested fix.",
            args_schema=CodeInput),
        StructuredTool.from_function(func=search_vehicle_manual(vector_store), name="search_vehicle_manual",
            description="Search uploaded vehicle manuals for relevant sections.", args_schema=QueryInput),
        StructuredTool.from_function(func=explore_related_codes, name="explore_related_codes",
            description="BFS — find all fault codes related to a given code within 2 hops.",
            args_schema=CodeSingleInput),
        StructuredTool.from_function(func=find_root_cause_chain, name="find_root_cause_chain",
            description="DFS — trace a fault code back to its deepest root cause.",
            args_schema=CodeSingleInput),
        StructuredTool.from_function(func=find_common_cause, name="find_common_cause",
            description="Find a single root cause explaining multiple reported fault codes.",
            args_schema=CodeInput),
        StructuredTool.from_function(func=predict_upcoming_codes, name="predict_next_codes",
            description="Predict which fault codes are likely to appear next if unresolved.",
            args_schema=CodeInput),
        StructuredTool.from_function(func=get_system_context, name="get_system_context",
            description="Find which vehicle system and subsystem a fault code belongs to.",
            args_schema=CodeSingleInput),
        StructuredTool.from_function(func=get_affected_components, name="get_affected_components",
            description="List all components potentially affected by a fault code.",
            args_schema=CodeSingleInput),
        StructuredTool.from_function(func=online_search, name="online_search",
            description="Search the web when a fault code isn't in the local database.",
            args_schema=QueryInput),
    ]


def get_safety_tools() -> list:
    return [
        StructuredTool.from_function(func=safety_assessment, name="safety_assessment",
            description="Assess safety severity (CRITICAL/HIGH/MEDIUM/clear) of reported symptoms.",
            args_schema=SymptomsInput),
        StructuredTool.from_function(func=fault_code_lookup, name="fault_code_lookup",
            description="Look up OBD-II fault codes for safety-relevance.", args_schema=CodeInput),
        StructuredTool.from_function(func=ask_human, name="ask_human",
            description="Ask a human for clarification on an ambiguous safety case.", args_schema=QueryInput),
    ]


def get_cost_tools() -> list:
    return [
        StructuredTool.from_function(func=repair_estimate, name="repair_estimate",
            description="Estimate repair costs for given components.", args_schema=ComponentsInput),
        StructuredTool.from_function(func=run_diagnostic_code, name="run_diagnostic_code",
            description="Run sandboxed Python for cost calculations.", args_schema=CodeExecInput),
        StructuredTool.from_function(func=get_repair_priority_tool, name="get_repair_priority",
            description="Order multiple fault codes by repair priority (severity-based).",
            args_schema=CodeInput),
        StructuredTool.from_function(func=get_repair_order, name="get_repair_order",
            description="Topologically sorted repair order respecting dependencies.",
            args_schema=CodeInput),
        StructuredTool.from_function(func=get_maintenance_schedule, name="maintenance_schedule",
            description="Get make-specific maintenance schedule.", args_schema=VehicleInput),
        StructuredTool.from_function(func=find_cheapest_path, name="find_cheapest_path",
            description="Dijkstra — find the minimum-cost diagnostic/repair path between two related fault codes.",
            args_schema=TwoCodeInput),
    ]