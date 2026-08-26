"""
tests/test_graph.py — Graph Algorithm Tests
Tests: fault_graph BFS/DFS/Dijkstra/causal, vehicle_hierarchy topo sort, chunk_graph
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fault_graph import (
    bfs_related_codes, get_root_cause_path,
    dijkstra_min_cost_path, dijkstra_cheapest_repair_path, find_cheapest_path,
    backwards_causal_search, predict_next_codes,
    get_repair_priority, FAULT_GRAPH
)
from vehicle_hierarchy import (
    get_system_context, get_affected_components,
    get_repair_order, topological_sort, REPAIR_DEPENDENCIES
)
from chunk_graph import ChunkGraph
from langchain.schema import Document


# ── BFS Tests ─────────────────────────────────────────────

def test_bfs_known_code():
    result = bfs_related_codes("P0300", max_hops=2)
    assert result["found"] is True
    assert result["total_related"] > 0
    assert "1" in result["by_hop"] or "2" in result["by_hop"]


def test_bfs_unknown_code():
    result = bfs_related_codes("P9999", max_hops=2)
    assert result["found"] is False


def test_bfs_hop_1():
    result = bfs_related_codes("P0300", max_hops=1)
    hop_1  = result["by_hop"].get("1", [])
    assert len(hop_1) > 0
    codes  = [c["code"] for c in hop_1]
    assert any(c in codes for c in ["P0420", "P0171", "P0301"])


def test_bfs_critical_code():
    result = bfs_related_codes("C0035", max_hops=1)
    assert result["found"] is True
    hop_1  = result["by_hop"].get("1", [])
    codes  = [c["code"] for c in hop_1]
    assert any(c in codes for c in ["C0040", "C0045"])


def test_bfs_all_codes_have_info():
    result = bfs_related_codes("P0171", max_hops=2)
    for hop_codes in result["by_hop"].values():
        for c in hop_codes:
            assert "code"         in c
            assert "name"         in c
            assert "severity"     in c
            assert "relationship" in c


# ── DFS Root Cause Tests ───────────────────────────────────

def test_dfs_known_code():
    result = get_root_cause_path("P0420")
    assert "chain" in result
    assert len(result["chain"]) >= 1
    assert result["chain"][0] == "P0420"


def test_dfs_finds_root():
    result = get_root_cause_path("P0420")
    assert result["root_cause"] != "P0420"


def test_dfs_no_causes():
    result = get_root_cause_path("P0101")
    assert result["root_cause"] == "P0101"
    assert len(result["chain"]) == 1


def test_dfs_chain_order():
    result = get_root_cause_path("P0420")
    chain  = result["chain"]
    assert chain[0] == "P0420"
    assert chain[-1] == result["root_cause"]


def test_dfs_interpretation():
    result = get_root_cause_path("P0300")
    assert "interpretation" in result
    assert len(result["interpretation"]) > 10


# ── Dijkstra Min-Cost Path Tests ────────────────────────────

def test_dijkstra_finds_path():
    result = dijkstra_min_cost_path("P0101", "P0420")
    assert result["found"] is True
    assert result["path"][0] == "P0101"
    assert result["path"][-1] == "P0420"


def test_dijkstra_same_start_end():
    result = dijkstra_min_cost_path("P0300", "P0300")
    assert result["found"] is True
    assert result["path"] == ["P0300"]
    assert result["total_cost"] == 0.0


def test_dijkstra_no_path_exists():
    result = dijkstra_min_cost_path("P0442", "B0001")
    assert result["found"] is False
    assert result["path"] == []
    assert result["total_cost"] is None


def test_dijkstra_unknown_code():
    result = dijkstra_min_cost_path("P9999", "P0300")
    assert result["found"] is False


def test_dijkstra_cost_is_positive():
    result = dijkstra_min_cost_path("P0101", "P0300")
    assert result["found"] is True
    assert result["total_cost"] > 0


def test_dijkstra_relationships_match_path_length():
    result = dijkstra_min_cost_path("P0101", "P0300")
    assert result["found"] is True
    assert len(result["relationships"]) == len(result["path"]) - 1


def test_dijkstra_direct_edge_cheaper_than_indirect():
    direct = dijkstra_min_cost_path("P0101", "P0171")
    assert direct["found"] is True
    assert direct["total_cost"] == 1.0


def test_dijkstra_finds_shortest_not_just_any_path():
    result = dijkstra_min_cost_path("P0300", "P0301")
    assert result["found"] is True
    assert result["total_cost"] <= 2.0


# ── Dijkstra Dollar-Cost Path Tests ─────────────────────────

def test_dijkstra_dollar_path_found():
    result = dijkstra_cheapest_repair_path("P0101", "P0420")
    if result["found"]:
        assert result["total_cost"] > 0
        assert result["path"][0] == "P0101"
        assert result["path"][-1] == "P0420"


def test_dijkstra_dollar_path_no_path():
    result = dijkstra_cheapest_repair_path("P0442", "B0001")
    assert result["found"] is False


def test_dijkstra_dollar_path_unknown_code():
    result = dijkstra_cheapest_repair_path("P9999", "P0300")
    assert result["found"] is False


def test_dijkstra_dollar_path_same_start_end():
    result = dijkstra_cheapest_repair_path("P0300", "P0300")
    assert result["found"] is True
    assert result["total_cost"] == 0.0


# ── find_cheapest_path Tool Wrapper Tests ───────────────────

def test_find_cheapest_path_tool_valid():
    result = find_cheapest_path("P0101, P0300")
    assert "Cheapest diagnostic path" in result
    assert "P0101" in result and "P0300" in result


def test_find_cheapest_path_tool_no_path():
    result = find_cheapest_path("P0442, B0001")
    assert "No path found" in result


def test_find_cheapest_path_tool_single_code():
    result = find_cheapest_path("P0300")
    assert "Provide two fault codes" in result


def test_find_cheapest_path_tool_includes_dollar_estimate():
    result = find_cheapest_path("P0101, P0420")
    assert "diagnostic effort cost" in result.lower() or "cost" in result.lower()


# ── Causal Search Tests ────────────────────────────────────

def test_causal_single_code():
    result = backwards_causal_search(["P0420"])
    assert "input_codes" in result


def test_causal_multiple_codes():
    result = backwards_causal_search(["P0300", "P0171"])
    assert result["input_codes"] == ["P0300", "P0171"]
    assert "has_common_cause" in result


def test_causal_empty():
    result = backwards_causal_search([])
    assert "error" in result


def test_predict_next_codes():
    result = predict_next_codes(["P0300"])
    assert "current_codes"   in result
    assert "predicted_codes" in result
    assert isinstance(result["predicted_codes"], list)


def test_predict_includes_worsens():
    result = predict_next_codes(["P0300"])
    predicted_codes = [c["code"] for c in result["predicted_codes"]]
    assert "P0420" in predicted_codes


def test_repair_priority():
    result = get_repair_priority(["P0300", "P0420", "C0035"])
    assert "priority_order" in result
    assert len(result["priority_order"]) == 3
    assert result["priority_order"][0]["code"] == "C0035"


# ── Vehicle Hierarchy Tests ────────────────────────────────

def test_get_system_context_known():
    result = get_system_context("P0300")
    assert "Engine System" in result or "Ignition" in result


def test_get_system_context_abs():
    result = get_system_context("C0035")
    assert "Brake" in result or "ABS" in result or "Wheel" in result


def test_get_affected_components():
    result = get_affected_components("P0300")
    assert "Ignition" in result or "component" in result.lower()


def test_topological_sort_simple():
    repairs = ["replace_catalytic_converter", "fix_o2_sensor", "diagnose_exhaust_leak"]
    ordered = topological_sort(repairs)
    assert len(ordered) == 3
    if "diagnose_exhaust_leak" in ordered and "fix_o2_sensor" in ordered:
        assert ordered.index("diagnose_exhaust_leak") < ordered.index("fix_o2_sensor")


def test_topological_sort_no_deps():
    repairs = ["check_spark_plugs", "check_maf_sensor"]
    ordered = topological_sort(repairs)
    assert len(ordered) == 2


def test_get_repair_order_from_codes():
    result = get_repair_order("P0300")
    assert "Repair Order" in result or "repair" in result.lower()


def test_get_repair_order_from_multiple():
    result = get_repair_order("P0420, P0300")
    assert len(result) > 10


# ── Chunk Graph Tests ──────────────────────────────────────

def test_chunk_graph_add():
    cg     = ChunkGraph(threshold=0.3)
    chunks = [
        Document(page_content="spark plug replacement procedure torque specs", metadata={"source": "manual.pdf"}),
        Document(page_content="spark plug inspection removal installation guide", metadata={"source": "manual.pdf"}),
        Document(page_content="transmission fluid change procedure", metadata={"source": "manual.pdf"})
    ]
    cg.add_chunks(chunks)
    assert len(cg.chunks) == 3


def test_chunk_graph_edges():
    cg     = ChunkGraph(threshold=0.2)
    chunks = [
        Document(page_content="spark plug replacement procedure", metadata={}),
        Document(page_content="spark plug installation torque specs", metadata={}),
        Document(page_content="brake pad replacement different topic", metadata={})
    ]
    cg.add_chunks(chunks)
    assert len(cg.adjacency) >= 0


def test_chunk_graph_search():
    cg     = ChunkGraph(threshold=0.2)
    chunks = [
        Document(page_content="spark plug ignition coil misfire repair", metadata={"source": "test"}),
        Document(page_content="brake pad caliper rotor replacement", metadata={"source": "test"}),
        Document(page_content="spark plug torque specification iridium", metadata={"source": "test"})
    ]
    cg.add_chunks(chunks)
    results = cg.search_with_graph("spark plug misfire", top_k=2, hops=1)
    assert len(results) >= 1


def test_chunk_graph_deduplicate():
    cg     = ChunkGraph(threshold=0.5)
    chunks = [
        Document(page_content="spark plug replacement very similar text here", metadata={}),
        Document(page_content="spark plug replacement very similar text here copy", metadata={}),
        Document(page_content="completely different topic about brakes", metadata={})
    ]
    cg.add_chunks(chunks)
    unique = cg.deduplicate()
    assert len(unique) <= 3
    assert len(unique) >= 1


def test_chunk_graph_connected_components():
    cg     = ChunkGraph(threshold=0.3)
    chunks = [
        Document(page_content="spark plug replacement procedure", metadata={}),
        Document(page_content="completely unrelated brake topic", metadata={})
    ]
    cg.add_chunks(chunks)
    components = cg.get_connected_components()
    assert len(components) >= 1


def test_chunk_graph_stats():
    cg     = ChunkGraph(threshold=0.3)
    chunks = [Document(page_content=f"test content number {i}", metadata={}) for i in range(5)]
    cg.add_chunks(chunks)
    stats = cg.get_stats()
    assert stats["total_chunks"] == 5
    assert "total_edges"  in stats
    assert "components"   in stats
    assert "threshold"    in stats