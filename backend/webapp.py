import os
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI()

# ── Env loader (no python-dotenv needed) ───────────────────
def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
load_env()


# ══════════════════════════════════════════════════════════
# REAL PROJECT LOGIC — inlined, no external deps, no internet
# ══════════════════════════════════════════════════════════

FAULT_CODES = {
    "P0101": {"name": "MAF Sensor Circuit Range/Performance", "severity": "MEDIUM",
        "causes": ["dirty MAF sensor", "vacuum leak", "wiring fault"],
        "urgency": "Schedule service within 2 weeks", "suggested_fix": "Clean or replace MAF sensor"},
    "P0171": {"name": "System Too Lean (Bank 1)", "severity": "MEDIUM",
        "causes": ["MAF sensor", "O2 sensor", "vacuum leak", "fuel pressure"],
        "urgency": "Schedule service within 2 weeks", "suggested_fix": "Inspect vacuum lines and MAF sensor"},
    "P0300": {"name": "Random/Multiple Cylinder Misfire", "severity": "HIGH",
        "causes": ["spark plugs", "ignition coils", "fuel injectors", "vacuum leak"],
        "urgency": "Address within days — risk of catalytic converter damage",
        "suggested_fix": "Replace spark plugs and inspect ignition coils"},
    "P0420": {"name": "Catalyst System Efficiency Below Threshold", "severity": "MEDIUM",
        "causes": ["faulty catalytic converter", "O2 sensors", "exhaust leaks"],
        "urgency": "Schedule service within 30 days",
        "suggested_fix": "Check O2 sensors first, replace catalytic converter if needed"},
    "P0700": {"name": "Transmission Control System Malfunction", "severity": "HIGH",
        "causes": ["low transmission fluid", "solenoids", "TCM failure"],
        "urgency": "Address promptly — risk of transmission damage",
        "suggested_fix": "Check transmission fluid level and condition first"},
    "C0035": {"name": "Left Front Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "causes": ["wheel speed sensor", "wiring", "ABS module"],
        "urgency": "Do not drive — immediate professional inspection required",
        "suggested_fix": "Replace left front wheel speed sensor"},
    "B0001": {"name": "Airbag Deployment Loop Fault", "severity": "CRITICAL",
        "causes": ["airbag sensor", "clock spring", "SRS module"],
        "urgency": "Do not drive immediately — safety system compromised",
        "suggested_fix": "Professional SRS system diagnosis required"},
}

SAFETY_TIERS = {
    "CRITICAL": ["fire", "smoke", "no brakes", "brake failure", "airbag", "steering failure",
                 "engine fire", "smell of gas", "fuel leak"],
    "HIGH":     ["abs light", "brake pedal", "brake noise", "soft brake", "power steering"],
    "MEDIUM":   ["traction control", "stability control", "tpms"],
}

REPAIR_COSTS = {
    "spark plugs": {"min": 100, "max": 300}, "ignition coils": {"min": 200, "max": 500},
    "o2 sensor": {"min": 150, "max": 400}, "maf sensor": {"min": 100, "max": 350},
    "catalytic converter": {"min": 500, "max": 2000}, "wheel speed sensor": {"min": 150, "max": 400},
    "transmission fluid": {"min": 100, "max": 250},
}

FAULT_GRAPH = {
    "P0101": {"causes": ["P0171"], "caused_by": []},
    "P0171": {"causes": ["P0300"], "caused_by": ["P0101"]},
    "P0300": {"causes": ["P0420"], "caused_by": ["P0171"]},
    "P0420": {"causes": [], "caused_by": ["P0300"]},
}

REPAIR_DEPENDENCIES = {
    "fix_maf_sensor": [], "fix_fuel_lean": ["fix_maf_sensor"],
    "fix_misfires": [], "replace_catalytic_converter": ["fix_misfires"],
}
CODE_TO_REPAIR = {
    "P0101": "fix_maf_sensor", "P0171": "fix_fuel_lean",
    "P0300": "fix_misfires", "P0420": "replace_catalytic_converter",
}


def fault_code_lookup(codes):
    results = []
    for code in [c.strip().upper() for c in codes.replace(",", " ").split() if c.strip()]:
        if code in FAULT_CODES:
            info = FAULT_CODES[code]
            results.append(f"{code} — {info['name']} [{info['severity']}]: {info['urgency']}. Fix: {info['suggested_fix']}")
        else:
            results.append(f"{code} — not in demo database")
    return results


def safety_assessment(symptoms):
    s = symptoms.lower()
    for tier, keywords in SAFETY_TIERS.items():
        matched = [k for k in keywords if k in s]
        if matched:
            return {"risk_level": tier, "matched": matched}
    return {"risk_level": "LOW", "matched": []}


def repair_estimate(codes):
    causes = set()
    for code in [c.strip().upper() for c in codes.replace(",", " ").split() if c.strip()]:
        if code in FAULT_CODES:
            causes.update(FAULT_CODES[code]["causes"])
    total_min = total_max = 0
    lines = []
    for cause in causes:
        for key, cost in REPAIR_COSTS.items():
            if key in cause.lower():
                lines.append(f"{cause}: ${cost['min']}-${cost['max']}")
                total_min += cost["min"]; total_max += cost["max"]
    return {"lines": lines, "total": f"${total_min}-${total_max}" if lines else "N/A"}


def bfs_related(code, max_hops=2):
    visited = {code}; frontier = [code]; related = []
    for _ in range(max_hops):
        next_frontier = []
        for c in frontier:
            for n in FAULT_GRAPH.get(c, {}).get("causes", []) + FAULT_GRAPH.get(c, {}).get("caused_by", []):
                if n not in visited:
                    visited.add(n); next_frontier.append(n); related.append(n)
        frontier = next_frontier
    return related


def dfs_root_cause(code):
    chain = [code]; current = code
    while FAULT_GRAPH.get(current, {}).get("caused_by"):
        current = FAULT_GRAPH[current]["caused_by"][0]
        chain.append(current)
    return chain


def topological_sort(repairs):
    ordered, visited = [], set()
    def visit(n):
        if n in visited: return
        visited.add(n)
        for dep in REPAIR_DEPENDENCIES.get(n, []): visit(dep)
        ordered.append(n)
    for r in repairs: visit(r)
    return ordered


def run_diagnosis(symptoms, fault_codes, vehicle_info):
    codes = [c.strip().upper() for c in fault_codes.replace(",", " ").split() if c.strip()]
    lookup = fault_code_lookup(fault_codes) if fault_codes else []
    safety = safety_assessment(symptoms)
    cost = repair_estimate(fault_codes) if fault_codes else {"lines": [], "total": "N/A"}
    repairs = [CODE_TO_REPAIR[c] for c in codes if c in CODE_TO_REPAIR]
    repair_order = topological_sort(repairs) if repairs else []
    related = bfs_related(codes[0]) if codes and codes[0] in FAULT_GRAPH else []
    root_chain = dfs_root_cause(codes[0]) if codes and codes[0] in FAULT_GRAPH else []

    return {
        "vehicle": vehicle_info or "Not specified",
        "symptoms": symptoms,
        "fault_code_lookup": lookup,
        "safety": safety,
        "cost_estimate": cost,
        "repair_order": repair_order,
        "bfs_related_codes": related,
        "dfs_root_cause_chain": root_chain,
        "timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════
# WEB UI
# ══════════════════════════════════════════════════════════

PAGE = """
<!DOCTYPE html><html><head><title>AutoDiag Pro — Demo</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; }
h2 { margin-bottom: 4px; }
.note { color: #666; font-size: 13px; margin-bottom: 20px; }
textarea, input { width: 100%; padding: 10px; margin: 6px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
button { padding: 10px 24px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
button:hover { background: #1d4ed8; }
#result { margin-top: 24px; }
.card { background: #f7f7f8; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
.critical { border-left: 4px solid #dc2626; }
.high { border-left: 4px solid #f59e0b; }
.medium { border-left: 4px solid #eab308; }
.low { border-left: 4px solid #16a34a; }
.label { font-weight: 600; font-size: 13px; text-transform: uppercase; color: #555; margin-bottom: 6px; }
</style></head>
<body>
<h2>🚗 AutoDiag Pro</h2>
<div class="note">Rule-based demo mode — the full version runs 4 coordinated LLM agents via LangChain/LangGraph.</div>

<textarea id="symptoms" rows="3" placeholder="Describe symptoms (e.g. rough idle, check engine light on)"></textarea>
<input id="codes" placeholder="Fault codes, optional (e.g. P0300 or P0101, P0300)">
<input id="vehicle" placeholder="Vehicle info, optional (e.g. Toyota Camry 2019)">
<button onclick="diagnose()">Diagnose</button>

<div id="result"></div>

<script>
async function diagnose() {
  document.getElementById('result').innerHTML = '<div class="card">Running diagnosis...</div>';
  const res = await fetch('/diagnose', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      symptoms: document.getElementById('symptoms').value,
      fault_codes: document.getElementById('codes').value,
      vehicle_info: document.getElementById('vehicle').value
    })
  });
  const d = await res.json();
  const riskClass = (d.safety.risk_level || 'low').toLowerCase();

  let html = '';
  html += `<div class="card ${riskClass}"><div class="label">Safety — ${d.safety.risk_level}</div>` +
          (d.safety.matched.length ? 'Detected: ' + d.safety.matched.join(', ') : 'No safety keywords detected') + '</div>';

  if (d.fault_code_lookup.length) {
    html += '<div class="card"><div class="label">Fault Codes</div>' + d.fault_code_lookup.join('<br>') + '</div>';
  }
  if (d.cost_estimate.lines.length) {
    html += '<div class="card"><div class="label">Cost Estimate</div>' + d.cost_estimate.lines.join('<br>') +
            '<br><b>Total: ' + d.cost_estimate.total + '</b></div>';
  }
  if (d.repair_order.length) {
    html += '<div class="card"><div class="label">Repair Order (dependency-sorted)</div>' + d.repair_order.join(' → ') + '</div>';
  }
  if (d.bfs_related_codes.length) {
    html += '<div class="card"><div class="label">Related Codes (BFS)</div>' + d.bfs_related_codes.join(', ') + '</div>';
  }
  if (d.dfs_root_cause_chain.length > 1) {
    html += '<div class="card"><div class="label">Root Cause Chain (DFS)</div>' + d.dfs_root_cause_chain.join(' → ') + '</div>';
  }
  document.getElementById('result').innerHTML = html;
}
</script>
</body></html>
"""


class DiagnoseRequest(BaseModel):
    symptoms: str = ""
    fault_codes: str = ""
    vehicle_info: str = ""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    return run_diagnosis(req.symptoms, req.fault_codes, req.vehicle_info)