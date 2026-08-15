"""
run_mission.py — Terminal Mission Runner (Updated)

New modes:
- Stream mode — results appear as each agent completes
- Graph demo — show BFS/DFS results visually
- Memory inspection — show what system remembers
- Resume mode — continue interrupted session
"""

import requests
import json
import argparse
import time
import sys

BASE_URL = "http://localhost:8000"


class C:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    PURPLE = "\033[95m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def ok(text):   print(f"{C.GREEN}✅ {text}{C.RESET}")
def warn(text): print(f"{C.YELLOW}⚠️  {text}{C.RESET}")
def err(text):  print(f"{C.RED}❌ {text}{C.RESET}")
def info(text): print(f"{C.BLUE}ℹ️  {text}{C.RESET}")


MISSIONS = {
    "1": {"name": "🔥 Cylinder Misfire",      "symptoms": "rough idle engine shaking poor acceleration check engine light on", "fault_codes": "P0300", "vehicle_info": "Toyota Camry 2019"},
    "2": {"name": "🛑 ABS Safety Critical",   "symptoms": "ABS light on brake pedal soft traction control warning",           "fault_codes": "C0035", "vehicle_info": "Honda Civic 2020"},
    "3": {"name": "💨 Engine Running Lean",    "symptoms": "poor idle hesitation acceleration fuel smell check engine",        "fault_codes": "P0171", "vehicle_info": "Ford Focus 2018"},
    "4": {"name": "🔧 Catalytic Converter",   "symptoms": "check engine light failed emissions test slight power loss",       "fault_codes": "P0420", "vehicle_info": "BMW 3 Series 2017"},
    "5": {"name": "⚡ Transmission Fault",    "symptoms": "transmission slipping harsh shifting wont shift third gear",       "fault_codes": "P0700", "vehicle_info": "Nissan Altima 2016"},
    "6": {"name": "🎯 Airbag (CRITICAL)",     "symptoms": "airbag warning light on dashboard",                               "fault_codes": "B0001", "vehicle_info": "Toyota Corolla 2021"},
    "7": {"name": "❓ No Fault Codes",        "symptoms": "car hesitates when cold stalls at low speed rough idle morning",  "fault_codes": "",      "vehicle_info": "Honda Accord 2015"},
    "8": {"name": "🔗 Multiple Codes (BFS)",  "symptoms": "rough idle poor economy check engine failed emissions",           "fault_codes": "P0300, P0171", "vehicle_info": "Ford Focus 2018"},
    "9": {"name": "🌳 Root Cause (DFS)",      "symptoms": "check engine catalytic converter warning",                        "fault_codes": "P0420", "vehicle_info": "BMW 2016"},
}


def check_server():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            ok("AutoDiag Pro is running")
            return True
    except Exception:
        pass
    err("Server not running. Start: uvicorn main:app --reload --port 8000")
    return False


def run_diagnosis(symptoms, fault_codes, vehicle_info, arch="master_agent", session_id=None):
    payload = {"symptoms": symptoms, "fault_codes": fault_codes, "vehicle_info": vehicle_info}
    if session_id:
        payload["session_id"] = session_id

    url = f"{BASE_URL}/diagnose/architecture/{arch}"
    print(f"\n{C.PURPLE}Sending to: {arch}{C.RESET}")

    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    done    = [False]

    def spin():
        i = 0
        while not done[0]:
            print(f"\r{C.YELLOW}Running agents {spinner[i % len(spinner)]}{C.RESET}", end="", flush=True)
            time.sleep(0.1)
            i += 1

    import threading
    t = threading.Thread(target=spin)
    t.start()
    start = time.time()

    try:
        r       = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start
        done[0] = True
        t.join()
        print(f"\r{C.GREEN}Completed in {elapsed:.1f}s              {C.RESET}")

        if r.status_code == 200:
            return r.json()
        else:
            err(f"API error {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        done[0] = True
        err(f"Error: {e}")
        return None


def run_stream_mode(symptoms, fault_codes, vehicle_info):
    """Stream mode — show results as each agent completes."""
    print(f"\n{C.CYAN}Stream mode — results appear as agents finish:{C.RESET}\n")

    payload = {"symptoms": symptoms, "fault_codes": fault_codes, "vehicle_info": vehicle_info}

    try:
        with requests.post(f"{BASE_URL}/diagnose/stream", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        if data_str == "[DONE]":
                            ok("Stream complete")
                            break
                        try:
                            chunk = json.loads(data_str)
                            node  = list(chunk.keys())[0] if chunk else "unknown"
                            print(f"  {C.YELLOW}→ {node}{C.RESET} completed")
                        except Exception:
                            pass
    except Exception as e:
        err(f"Stream error: {e}")


def run_graph_demo(fault_code="P0300"):
    """Graph demo — show BFS/DFS results visually."""
    print(f"\n{C.BOLD}{C.CYAN}Graph Algorithm Demo for {fault_code}{C.RESET}")
    print(f"{C.CYAN}{'='*50}{C.RESET}")

    # BFS
    print(f"\n{C.YELLOW}BFS — Related Codes (2 hops):{C.RESET}")
    try:
        r = requests.get(f"{BASE_URL}/graph/fault/related/{fault_code}?max_hops=2", timeout=10)
        if r.status_code == 200:
            data = r.json()
            for hop, codes in data.get("by_hop", {}).items():
                print(f"  Hop {hop}:")
                for c in codes[:4]:
                    print(f"    {C.GREEN}{c['code']}{C.RESET} — {c['name']} [{c['severity']}] via {c['relationship']}")
        else:
            warn("BFS endpoint not available")
    except Exception as e:
        warn(f"BFS error: {e}")

    # DFS Root Cause
    print(f"\n{C.YELLOW}DFS — Root Cause Chain:{C.RESET}")
    try:
        r = requests.get(f"{BASE_URL}/graph/fault/root-cause/{fault_code}", timeout=10)
        if r.status_code == 200:
            data  = r.json()
            chain = " → ".join(data.get("chain", [fault_code]))
            print(f"  Chain: {C.GREEN}{chain}{C.RESET}")
            print(f"  Root:  {C.RED}{data.get('root_cause','')}{C.RESET} — {data.get('root_name','')}")
            print(f"  {data.get('interpretation','')}")
        else:
            warn("DFS endpoint not available")
    except Exception as e:
        warn(f"DFS error: {e}")

    # Repair Order
    print(f"\n{C.YELLOW}Topological Sort — Repair Order:{C.RESET}")
    try:
        r = requests.post(f"{BASE_URL}/graph/repair-order", json=[fault_code], timeout=10)
        if r.status_code == 200:
            data = r.json()
            order = data.get("repair_order", "")
            for line in str(order).split("\n")[:6]:
                if line.strip():
                    print(f"  {line}")
        else:
            warn("Repair order endpoint not available")
    except Exception as e:
        warn(f"Topo sort error: {e}")


def run_memory_inspection(session_id):
    """Show what system remembers about a session."""
    print(f"\n{C.BOLD}{C.CYAN}Memory Inspection — Session: {session_id}{C.RESET}")
    print(f"{C.CYAN}{'='*50}{C.RESET}")

    try:
        r = requests.get(f"{BASE_URL}/memory/{session_id}", timeout=10)
        if r.status_code == 200:
            stats = r.json()
            print(f"  Episodic memories:  {stats.get('episodic_count', 0)}")
            print(f"  Semantic facts:     {stats.get('semantic_count', 0)}")
            print(f"  Saved procedures:   {stats.get('procedural_count', 0)}")
            print(f"  Has short-term:     {stats.get('has_short_term', False)}")
        else:
            warn(f"Memory endpoint error: {r.status_code}")
    except Exception as e:
        warn(f"Memory inspection error: {e}")

    # DB stats
    try:
        r = requests.get(f"{BASE_URL}/admin/db-stats", headers={"X-API-Key": "dev-key-123"}, timeout=10)
        if r.status_code == 200:
            stats = r.json()
            print(f"\n  SQLite DB Stats:")
            for k, v in stats.items():
                print(f"    {k}: {v}")
    except Exception:
        pass


def print_result(result, arch):
    if not result:
        return

    print(f"\n{C.BOLD}{C.CYAN}{'='*60}")
    print(f"  AUTODIAG PRO — DIAGNOSTIC REPORT")
    print(f"  Architecture: {arch.upper()}")
    print(f"{'='*60}{C.RESET}")

    # Safety first
    safety = result.get("safety", {})
    if safety:
        risk = safety.get("risk_level", "UNKNOWN")
        safe = safety.get("is_safe_to_drive", True)
        if risk == "CRITICAL":
            print(f"\n  {C.RED}{C.BOLD}⛔ CRITICAL — DO NOT DRIVE{C.RESET}")
        elif risk == "HIGH":
            print(f"\n  {C.YELLOW}⚠️  HIGH RISK — Schedule immediate repair{C.RESET}")
        elif risk == "MEDIUM":
            print(f"\n  {C.YELLOW}🔶 MEDIUM — Schedule repair within 1 week{C.RESET}")
        else:
            print(f"\n  {C.GREEN}✅ LOW RISK — Safe to drive{C.RESET}")

    if result.get("needs_human_review"):
        print(f"\n  {C.RED}{C.BOLD}👨‍🔧 HUMAN REVIEW REQUIRED{C.RESET}")

    # Diagnosis
    diagnosis = result.get("diagnosis", {})
    if diagnosis:
        print(f"\n{C.YELLOW}── Diagnosis ──{C.RESET}")
        primary = diagnosis.get("primary_issue", "")

        primary = diagnosis.get("primary_issue", "")
        if primary:
            print(f"  {str(primary)[:300]}")

        causes = diagnosis.get("likely_causes", [])
        if causes:
            print(f"\n  Likely Causes: {', '.join(causes[:5])}")

        root_chain = diagnosis.get("root_cause_chain", [])
        if root_chain:
            print(f"  Root Cause Chain: {' → '.join(root_chain[:4])}")

        related = diagnosis.get("related_codes", [])
        if related:
            print(f"  Related Codes: {', '.join(related[:5])}")

        predicted = diagnosis.get("predicted_codes", [])
        if predicted:
            print(f"  {C.YELLOW}⚠️  Predicted upcoming: {', '.join(predicted[:3])}{C.RESET}")

    # Cost
    cost = result.get("cost_estimate", {})
    if cost:
        print(f"\n{C.YELLOW}── Cost Estimate ──{C.RESET}")
        estimate = cost.get("estimate", "")
        if estimate:
            print(f"  {str(estimate)[:200]}")

    # Reflection
    reflection = result.get("reflection", "")
    if isinstance(reflection, dict):
        reflection = reflection.get("final_answer", "")
    if reflection:
        print(f"\n{C.YELLOW}── Reflection ──{C.RESET}")
        print(f"  {str(reflection)[:200]}")

    # Graph of Thoughts
    got = result.get("got_reasoning", {})
    if got and got.get("merges_occurred", 0) > 0:
        print(f"\n{C.YELLOW}── Graph of Thoughts ──{C.RESET}")
        print(f"  Nodes explored: {got.get('nodes_explored', 0)}")
        print(f"  Branch merges:  {got.get('merges_occurred', 0)}")
        print(f"  Best score:     {got.get('best_score', 0):.2f}")

    # Trace
    trace = result.get("trace_summary", {})
    if trace:
        print(f"\n{C.YELLOW}── Execution ──{C.RESET}")
        print(f"  Steps:    {trace.get('total_steps', 'N/A')}")
        print(f"  Cost:     ${trace.get('total_cost_usd', 0):.4f}")
        redundant = trace.get("redundant_calls", [])
        if redundant:
            warn(f"Redundant calls: {len(redundant)}")
        else:
            ok("No redundant calls")

    timing = result.get("total_time") or result.get("api_latency", "N/A")
    print(f"\n  {C.CYAN}Total time: {timing}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'='*60}{C.RESET}\n")


def select_mission():
    print(f"\n{C.BOLD}{C.CYAN}AutoDiag Pro v2 — Mission Selector{C.RESET}")
    print(f"{C.CYAN}{'='*40}{C.RESET}\n")

    for key, m in MISSIONS.items():
        print(f"  {C.YELLOW}[{key}]{C.RESET} {m['name']}")
        if m['fault_codes']:
            print(f"       Codes: {m['fault_codes']}")

    print(f"\n  {C.YELLOW}[C]{C.RESET} Custom mission")
    print(f"  {C.YELLOW}[G]{C.RESET} Graph algorithm demo")
    print(f"  {C.YELLOW}[Q]{C.RESET} Quit\n")

    choice = input(f"{C.BOLD}Choice: {C.RESET}").strip().upper()

    if choice == "Q":
        print("Goodbye!")
        sys.exit(0)
    elif choice == "G":
        code = input("Fault code for demo [P0300]: ").strip() or "P0300"
        return None, None, None, None, "graph_demo", code
    elif choice == "C":
        symptoms     = input("Symptoms: ").strip()
        fault_codes  = input("Fault codes (Enter to skip): ").strip()
        vehicle_info = input("Vehicle info (Enter to skip): ").strip()
        return symptoms, fault_codes, vehicle_info, None, "diagnose", None
    elif choice in MISSIONS:
        m = MISSIONS[choice]
        return m["symptoms"], m["fault_codes"], m["vehicle_info"], None, "diagnose", None
    else:
        m = MISSIONS["1"]
        return m["symptoms"], m["fault_codes"], m["vehicle_info"], None, "diagnose", None


def select_architecture():
    archs = [
        ("master_agent",       "Orchestrator + GoT reasoning (recommended)"),
        ("sequential_pipeline","Assembly line A→B→C→D"),
        ("collaborative",      "All agents share full context"),
        ("parallel_swarm",     "All independent, results merged"),
        ("hierarchical",       "3-level: Chief → Leads → Workers"),
        ("blackboard_only",    "All via shared blackboard"),
        ("peer_to_peer",       "No orchestrator, direct comms"),
        ("self_improving",     "Iterates with self-critique"),
    ]

    print(f"\n{C.BOLD}Select Architecture:{C.RESET}")
    for i, (arch, desc) in enumerate(archs, 1):
        print(f"  {C.YELLOW}[{i}]{C.RESET} {arch:<25} {C.RESET}{desc}")

    choice = input(f"\n{C.BOLD}Architecture [1-8, default=1]: {C.RESET}").strip()

    if not choice:
        return archs[0][0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(archs):
            return archs[idx][0]
    except ValueError:
        pass
    return "master_agent"


def run_quick_test():
    print(f"\n{C.BOLD}{C.CYAN}AutoDiag Pro v2 — Quick Test{C.RESET}")
    result = run_diagnosis(
        "rough idle and check engine light on", "P0300", "Toyota Camry 2019", "master_agent"
    )
    print_result(result, "master_agent")


def run_compare_all():
    print(f"\n{C.BOLD}{C.CYAN}Comparing All 8 Architectures{C.RESET}\n")
    archs   = ["master_agent", "sequential_pipeline", "collaborative", "parallel_swarm",
               "hierarchical", "blackboard_only", "peer_to_peer", "self_improving"]
    results = {}

    for arch in archs:
        print(f"  Running {arch}...", end="", flush=True)
        start  = time.time()
        result = run_diagnosis("rough idle check engine P0300", "P0300", "Toyota 2019", arch)
        elapsed = time.time() - start

        if result:
            results[arch] = {"status": "✅", "time": f"{elapsed:.1f}s"}
            print(f"\r  {C.GREEN}✅ {arch:<30} {elapsed:.1f}s{C.RESET}")
        else:
            results[arch] = {"status": "❌", "time": f"{elapsed:.1f}s"}
            print(f"\r  {C.RED}❌ {arch:<30} failed{C.RESET}")

    print(f"\n{C.BOLD}{'='*50}{C.RESET}")
    for arch, r in results.items():
        color = C.GREEN if r["status"] == "✅" else C.RED
        print(f"  {arch:<30} {color}{r['status']}{C.RESET} {r['time']}")


def main():
    parser = argparse.ArgumentParser(description="AutoDiag Pro v2 Terminal Runner")
    parser.add_argument("--arch",     default=None)
    parser.add_argument("--symptoms", default=None)
    parser.add_argument("--codes",    default="")
    parser.add_argument("--vehicle",  default="")
    parser.add_argument("--quick",    action="store_true")
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--stream",   action="store_true")
    parser.add_argument("--graph",    default=None, help="Graph demo for fault code e.g. P0300")
    parser.add_argument("--memory",   default=None, help="Inspect memory for session_id")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.CYAN}")
    print("  ╔═════════════════════════════════════════╗")
    print("  ║    AutoDiag Pro v2 — Terminal Runner    ║")
    print("  ║  Multi-Agent + Graph Algorithms + GoT   ║")
    print("  ╚═════════════════════════════════════════╝")
    print(f"{C.RESET}")

    if not check_server():
        sys.exit(1)

    if args.quick:
        run_quick_test()
        return

    if args.compare:
        run_compare_all()
        return

    if args.graph:
        run_graph_demo(args.graph)
        return

    if args.memory:
        run_memory_inspection(args.memory)
        return

    if args.symptoms:
        arch   = args.arch or "master_agent"
        if args.stream:
            run_stream_mode(args.symptoms, args.codes, args.vehicle)
        else:
            result = run_diagnosis(args.symptoms, args.codes, args.vehicle, arch)
            print_result(result, arch)
        return

    # Interactive mode
    while True:
        symptoms, fault_codes, vehicle_info, session_id, mode, extra = select_mission()

        if mode == "graph_demo":
            run_graph_demo(extra or "P0300")
        else:
            arch = select_architecture()

            use_stream = input(f"\n{C.BOLD}Stream mode? [y/N]: {C.RESET}").strip().lower() == "y"

            if use_stream:
                run_stream_mode(symptoms, fault_codes, vehicle_info)
            else:
                result = run_diagnosis(symptoms, fault_codes, vehicle_info, arch, session_id)
                print_result(result, arch)

                if result and result.get("session_id"):
                    show_mem = input(f"\n{C.BOLD}Inspect memory? [y/N]: {C.RESET}").strip().lower() == "y"
                    if show_mem:
                        run_memory_inspection(result["session_id"])

        again = input(f"\n{C.BOLD}Run another? [y/N]: {C.RESET}").strip().lower()
        if again != "y":
            print(f"\n{C.CYAN}Goodbye! Check MLflow at http://localhost:5000{C.RESET}\n")
            break


if __name__ == "__main__":
    main()

