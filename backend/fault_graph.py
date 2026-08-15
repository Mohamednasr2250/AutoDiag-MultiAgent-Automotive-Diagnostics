"""
fault_graph.py — Fault Code Relationship Graph
Nodes = OBD-II fault codes. Edge types: causes, caused_by, related_to, worsens, co_occurs.
BFS (related codes), DFS (root cause), Dijkstra (min-cost path), backwards causal
search, prediction, repair priority.
"""

import heapq
from typing import List, Dict, Any

FAULT_GRAPH: Dict[str, Dict[str, Any]] = {
    "P0101": {"name": "MAF Sensor Circuit Range/Performance", "severity": "MEDIUM",
        "causes": ["P0171"], "caused_by": [], "related_to": ["P0102", "P0103"],
        "worsens": [], "co_occurs": ["P0171"]},
    "P0171": {"name": "System Too Lean (Bank 1)", "severity": "MEDIUM",
        "causes": ["P0300"], "caused_by": ["P0101"], "related_to": ["P0174"],
        "worsens": ["P0300"], "co_occurs": ["P0300"]},
    "P0300": {"name": "Random/Multiple Cylinder Misfire", "severity": "HIGH",
        "causes": ["P0420"], "caused_by": ["P0171"],
        "related_to": ["P0301", "P0302", "P0303", "P0304"],
        "worsens": ["P0420"], "co_occurs": ["P0171"]},
    "P0301": {"name": "Cylinder 1 Misfire", "severity": "HIGH",
        "causes": [], "caused_by": ["P0300"], "related_to": ["P0300"], "worsens": [], "co_occurs": []},
    "P0302": {"name": "Cylinder 2 Misfire", "severity": "HIGH",
        "causes": [], "caused_by": ["P0300"], "related_to": ["P0300"], "worsens": [], "co_occurs": []},
    "P0303": {"name": "Cylinder 3 Misfire", "severity": "HIGH",
        "causes": [], "caused_by": ["P0300"], "related_to": ["P0300"], "worsens": [], "co_occurs": []},
    "P0304": {"name": "Cylinder 4 Misfire", "severity": "HIGH",
        "causes": [], "caused_by": ["P0300"], "related_to": ["P0300"], "worsens": [], "co_occurs": []},
    "P0420": {"name": "Catalyst System Efficiency Below Threshold", "severity": "MEDIUM",
        "causes": [], "caused_by": ["P0300"], "related_to": ["P0430"],
        "worsens": [], "co_occurs": ["P0300"]},
    "P0430": {"name": "Catalyst System Efficiency (Bank 2)", "severity": "MEDIUM",
        "causes": [], "caused_by": [], "related_to": ["P0420"], "worsens": [], "co_occurs": []},
    "P0442": {"name": "EVAP System Small Leak", "severity": "LOW",
        "causes": [], "caused_by": [], "related_to": ["P0455"], "worsens": [], "co_occurs": []},
    "P0455": {"name": "EVAP System Large Leak", "severity": "LOW",
        "causes": [], "caused_by": [], "related_to": ["P0442"], "worsens": [], "co_occurs": []},
    "P0700": {"name": "Transmission Control System Malfunction", "severity": "HIGH",
        "causes": [], "caused_by": [], "related_to": ["P0730"], "worsens": [], "co_occurs": []},
    "P0730": {"name": "Incorrect Gear Ratio", "severity": "HIGH",
        "causes": [], "caused_by": [], "related_to": ["P0700"], "worsens": [], "co_occurs": []},
    "C0035": {"name": "Left Front Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "causes": [], "caused_by": [], "related_to": ["C0040", "C0045"],
        "worsens": [], "co_occurs": ["C0040"]},
    "C0040": {"name": "Right Front Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "causes": [], "caused_by": [], "related_to": ["C0035", "C0045"], "worsens": [], "co_occurs": []},
    "C0045": {"name": "Rear Wheel Speed Sensor Circuit", "severity": "CRITICAL",
        "causes": [], "caused_by": [], "related_to": ["C0035", "C0040"], "worsens": [], "co_occurs": []},
    "B0001": {"name": "Airbag Deployment Loop Fault", "severity": "CRITICAL",
        "causes": [], "caused_by": [], "related_to": [], "worsens": [], "co_occurs": []},
    "U0100": {"name": "Lost Communication With ECM/PCM", "severity": "HIGH",
        "causes": [], "caused_by": [], "related_to": [], "worsens": [], "co_occurs": []},
}

EDGE_TYPES = ["causes", "caused_by", "related_to", "worsens", "co_occurs"]

EDGE_WEIGHTS = {
    "causes": 1.0,
    "caused_by": 1.0,
    "worsens": 1.5,
    "related_to": 2.0,
    "co_occurs": 2.5,
}


def _neighbors(code: str) -> List[tuple]:
    node = FAULT_GRAPH.get(code, {})
    out = []
    for edge_type in EDGE_TYPES:
        for n in node.get(edge_type, []):
            out.append((n, edge_type))
    return out


def bfs_related_codes(code: str, max_hops: int = 2) -> Dict[str, Any]:
    code = code.strip().upper()
    if code not in FAULT_GRAPH:
        return {"found": False, "code": code, "total_related": 0, "by_hop": {}}

    visited = {code}
    by_hop: Dict[str, List[Dict]] = {}
    frontier = [code]

    for hop in range(1, max_hops + 1):
        next_frontier = []
        hop_results = []
        for c in frontier:
            for neighbor, rel in _neighbors(c):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    info = FAULT_GRAPH.get(neighbor, {})
                    hop_results.append({
                        "code": neighbor, "name": info.get("name", "Unknown"),
                        "severity": info.get("severity", "UNKNOWN"), "relationship": rel
                    })
        if hop_results:
            by_hop[str(hop)] = hop_results
        frontier = next_frontier
        if not frontier:
            break

    total = sum(len(v) for v in by_hop.values())
    return {"found": True, "code": code, "total_related": total, "by_hop": by_hop}


def get_root_cause_path(code: str) -> Dict[str, Any]:
    code = code.strip().upper()
    chain = [code]
    visited = {code}
    current = code

    while True:
        node = FAULT_GRAPH.get(current, {})
        causes = [c for c in node.get("caused_by", []) if c not in visited]
        if not causes:
            break
        current = causes[0]
        chain.append(current)
        visited.add(current)

    root = chain[-1]
    root_name = FAULT_GRAPH.get(root, {}).get("name", "Unknown")

    if len(chain) == 1:
        interpretation = f"{code} appears to be the root cause itself — no further upstream cause found."
    else:
        interpretation = (
            f"Tracing backwards from {code}, the root cause is {root} ({root_name}). "
            f"Fixing {root} first should resolve the entire chain: {' -> '.join(chain)}."
        )

    return {"chain": chain, "root_cause": root, "root_name": root_name, "interpretation": interpretation}


def dijkstra_min_cost_path(start: str, end: str) -> Dict[str, Any]:
    """Minimum diagnostic-effort path between two fault codes."""
    start, end = start.strip().upper(), end.strip().upper()
    if start not in FAULT_GRAPH or end not in FAULT_GRAPH:
        return {"found": False, "path": [], "total_cost": None}

    if start == end:
        return {"found": True, "path": [start], "relationships": [], "total_cost": 0.0}

    distances = {start: 0.0}
    previous: Dict[str, tuple] = {}
    visited = set()
    heap = [(0.0, start)]

    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            break
        for neighbor, rel in _neighbors(node):
            weight = EDGE_WEIGHTS.get(rel, 2.0)
            new_dist = dist + weight
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = (node, rel)
                heapq.heappush(heap, (new_dist, neighbor))

    if end not in distances:
        return {"found": False, "path": [], "total_cost": None}

    path, relationships = [end], []
    current = end
    while current != start:
        prev_node, rel = previous[current]
        relationships.append(rel)
        path.append(prev_node)
        current = prev_node
    path.reverse()
    relationships.reverse()

    return {"found": True, "path": path, "relationships": relationships, "total_cost": round(distances[end], 2)}


def dijkstra_cheapest_repair_path(start: str, end: str) -> Dict[str, Any]:
    """Same traversal, weighted by actual repair dollar cost instead of relationship type."""
    from tools import REPAIR_COSTS, FAULT_CODES  # lazy import avoids circular import at module load

    def dollar_weight(code: str) -> float:
        causes = FAULT_CODES.get(code, {}).get("causes", [])
        costs = [REPAIR_COSTS[c]["min"] for c in causes if c in REPAIR_COSTS]
        return min(costs) if costs else 200.0

    start, end = start.strip().upper(), end.strip().upper()
    if start not in FAULT_GRAPH or end not in FAULT_GRAPH:
        return {"found": False, "path": [], "total_cost": None}

    if start == end:
        return {"found": True, "path": [start], "total_cost": 0.0}

    distances = {start: 0.0}
    previous: Dict[str, tuple] = {}
    visited = set()
    heap = [(0.0, start)]

    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            break
        for neighbor, rel in _neighbors(node):
            weight = dollar_weight(neighbor)
            new_dist = dist + weight
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = (node, rel)
                heapq.heappush(heap, (new_dist, neighbor))

    if end not in distances:
        return {"found": False, "path": [], "total_cost": None}

    path = [end]
    current = end
    while current != start:
        prev_node, _ = previous[current]
        path.append(prev_node)
        current = prev_node
    path.reverse()

    return {"found": True, "path": path, "total_cost": round(distances[end], 2)}


def backwards_causal_search(codes: List[str]) -> Dict[str, Any]:
    codes = [c.strip().upper() for c in codes if c.strip()]
    if not codes:
        return {"error": "No codes provided"}

    chains = {c: get_root_cause_path(c)["chain"] for c in codes}
    chain_sets = [set(v) for v in chains.values()]
    common = set.intersection(*chain_sets) if chain_sets else set()
    common_ancestors = [
        {"code": c, "name": FAULT_GRAPH.get(c, {}).get("name", "Unknown")} for c in common
    ]

    return {
        "input_codes": codes, "chains": chains,
        "common_ancestors": common_ancestors,
        "has_common_cause": len(common_ancestors) > 0
    }


def predict_next_codes(codes: List[str]) -> Dict[str, Any]:
    codes = [c.strip().upper() for c in codes if c.strip()]
    predicted = {}
    for c in codes:
        node = FAULT_GRAPH.get(c, {})
        for n in node.get("worsens", []) + node.get("co_occurs", []):
            if n not in codes:
                predicted[n] = FAULT_GRAPH.get(n, {})

    predicted_list = [
        {"code": c, "name": info.get("name", "Unknown"), "severity": info.get("severity", "UNKNOWN")}
        for c, info in predicted.items()
    ]
    return {"current_codes": codes, "predicted_codes": predicted_list}


def get_repair_priority(codes: List[str]) -> Dict[str, Any]:
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    codes = [c.strip().upper() for c in codes if c.strip()]
    items = [
        {"code": c, "name": FAULT_GRAPH.get(c, {}).get("name", "Unknown"),
         "severity": FAULT_GRAPH.get(c, {}).get("severity", "UNKNOWN")}
        for c in codes
    ]
    items.sort(key=lambda x: order.get(x["severity"], 4))
    return {"priority_order": items}


def explore_related_codes(code: str) -> str:
    result = bfs_related_codes(code, max_hops=2)
    if not result["found"]:
        return f"Code {code} not found in fault graph."
    lines = [f"Related codes for {code} ({result['total_related']} total):"]
    for hop, items in result["by_hop"].items():
        lines.append(f"Hop {hop}:")
        for item in items:
            lines.append(f"  - {item['code']} ({item['relationship']}): {item['name']} [{item['severity']}]")
    return "\n".join(lines)


def find_root_cause_chain(code: str) -> str:
    result = get_root_cause_path(code)
    return f"Root Cause Chain: {' -> '.join(result['chain'])}\n{result['interpretation']}"


def find_common_cause(codes_str: str) -> str:
    codes = [c.strip() for c in codes_str.replace(",", " ").split() if c.strip()]
    result = backwards_causal_search(codes)
    if "error" in result:
        return result["error"]
    if result["has_common_cause"]:
        names = ", ".join(a["code"] for a in result["common_ancestors"])
        return f"Common cause found for {codes}: {names}. Fixing this resolves all reported codes."
    return f"No single common cause found for {codes}. Each may need separate repair."


def predict_upcoming_codes(code: str) -> str:
    codes = [c.strip() for c in code.replace(",", " ").split() if c.strip()]
    result = predict_next_codes(codes)
    if not result["predicted_codes"]:
        return f"No predicted upcoming codes for {codes} based on current graph."
    lines = [f"Predicted upcoming codes for {codes}:"]
    for p in result["predicted_codes"]:
        lines.append(f"  - {p['code']}: {p['name']} [{p['severity']}]")
    return "\n".join(lines)


def get_repair_priority_tool(codes_str: str) -> str:
    codes = [c.strip() for c in codes_str.replace(",", " ").split() if c.strip()]
    result = get_repair_priority(codes)
    lines = ["Repair Priority Order:"]
    for i, item in enumerate(result["priority_order"], 1):
        lines.append(f"  {i}. {item['code']} [{item['severity']}] — {item['name']}")
    return "\n".join(lines)


def find_cheapest_path(codes_str: str) -> str:
    """Agent-facing Dijkstra tool: 'P0300, P0420' -> cheapest diagnostic + dollar path between them."""
    codes = [c.strip().upper() for c in codes_str.replace(",", " ").split() if c.strip()]
    if len(codes) < 2:
        return "Provide two fault codes separated by a comma, e.g. 'P0300, P0420'."

    result = dijkstra_min_cost_path(codes[0], codes[1])
    if not result["found"]:
        return f"No path found between {codes[0]} and {codes[1]} in the fault graph."

    dollar_result = dijkstra_cheapest_repair_path(codes[0], codes[1])
    lines = [
        f"Cheapest diagnostic path {codes[0]} -> {codes[1]}:",
        f"  Path: {' -> '.join(result['path'])}",
        f"  Relationships: {', '.join(result['relationships'])}",
        f"  Diagnostic effort cost: {result['total_cost']}",
    ]
    if dollar_result["found"]:
        lines.append(f"  Estimated dollar-cost path: {' -> '.join(dollar_result['path'])} (~${dollar_result['total_cost']})")
    return "\n".join(lines)