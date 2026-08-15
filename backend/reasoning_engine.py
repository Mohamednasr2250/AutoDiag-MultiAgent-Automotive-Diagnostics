"""
reasoning_engine.py — Tree of Thoughts / Graph of Thoughts + World Models

Note: Monte Carlo Tree Search (MCTS) was deliberately removed from this
engine. Its role — planning a tool-call sequence before execution — is
already covered more directly by the ReAct loop itself (agents choose
actions incrementally with real observations) and by the fault-code
Dijkstra pathing in fault_graph.py, which gives an exact cheapest-path
answer instead of a simulated/approximate one.
"""

import re
from typing import List, Dict, Any


class ThoughtNode:
    _id_counter = 0

    def __init__(self, content: str, parent: "ThoughtNode" = None, score: float = 0.0):
        ThoughtNode._id_counter += 1
        self.id = ThoughtNode._id_counter
        self.content = content
        self.parents: List["ThoughtNode"] = [parent] if parent else []
        self.children: List["ThoughtNode"] = []
        self.score = score
        self.merged = False

    def add_child(self, child: "ThoughtNode"):
        self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def merge_with(self, other: "ThoughtNode", merged_content: str) -> "ThoughtNode":
        merged = ThoughtNode(merged_content, score=max(self.score, other.score) + 0.1)
        merged.parents = [self, other]
        self.children.append(merged)
        other.children.append(merged)
        merged.merged = True
        return merged


class TreeOfThoughts:
    def __init__(self, llm, breadth: int = 3, depth: int = 2):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth
        self.all_nodes: List[ThoughtNode] = []

    def _generate_thoughts(self, context: str, n: int) -> List[str]:
        prompt = (f"Given this vehicle issue: {context}\n"
                  f"Generate {n} distinct possible diagnostic hypotheses, one per line.")
        try:
            response = self.llm.invoke(prompt)
            lines = [l.strip("- ").strip() for l in response.split("\n") if l.strip()]
            return lines[:n] if lines else [f"Hypothesis {i+1} for: {context}" for i in range(n)]
        except Exception:
            return [f"Hypothesis {i+1} for: {context}" for i in range(n)]

    def _score_thought(self, thought: str, context: str) -> float:
        prompt = f"Rate how plausible this diagnosis is for '{context}' on a 0-1 scale: {thought}\nScore:"
        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r"\d*\.?\d+", response)
            return min(1.0, float(numbers[0])) if numbers else 0.5
        except Exception:
            return 0.5

    def run(self, context: str) -> Dict[str, Any]:
        root = ThoughtNode(f"Root: {context}", score=1.0)
        self.all_nodes = [root]
        frontier = [root]

        for level in range(self.depth):
            next_frontier = []
            for node in frontier:
                thoughts = self._generate_thoughts(node.content, self.breadth)
                for t in thoughts:
                    score = self._score_thought(t, context)
                    child = ThoughtNode(t, parent=node, score=score)
                    node.add_child(child)
                    self.all_nodes.append(child)
                    next_frontier.append(child)
            next_frontier.sort(key=lambda n: n.score, reverse=True)
            frontier = next_frontier[:self.breadth]

        best = max(self.all_nodes, key=lambda n: n.score)
        return {"best_hypothesis": best.content, "best_score": best.score,
                "path": self._trace_path(best), "nodes_explored": len(self.all_nodes)}

    def _trace_path(self, node: ThoughtNode) -> List[str]:
        path = [node.content]
        current = node
        while current.parents:
            current = current.parents[0]
            path.append(current.content)
        return list(reversed(path))


class GraphOfThoughts(TreeOfThoughts):
    """Extends ToT — convergent branches (same conclusion) are merged, boosting confidence."""

    def _similarity(self, a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def merge_thoughts(self, nodes: List[ThoughtNode], threshold: float = 0.5) -> List[ThoughtNode]:
        merged_nodes = []
        used = set()
        for i, n1 in enumerate(nodes):
            if n1.id in used:
                continue
            cluster = [n1]
            for n2 in nodes[i + 1:]:
                if n2.id not in used and self._similarity(n1.content, n2.content) >= threshold:
                    cluster.append(n2)
                    used.add(n2.id)
            if len(cluster) > 1:
                merged_content = f"Convergent conclusion: {cluster[0].content}"
                merged = cluster[0]
                for other in cluster[1:]:
                    merged = merged.merge_with(other, merged_content)
                self.all_nodes.append(merged)
                merged_nodes.append(merged)
            else:
                merged_nodes.append(n1)
            used.add(n1.id)
        return merged_nodes

    def run(self, context: str) -> Dict[str, Any]:
        root = ThoughtNode(f"Root: {context}", score=1.0)
        self.all_nodes = [root]
        frontier = [root]

        for level in range(self.depth):
            next_frontier = []
            for node in frontier:
                thoughts = self._generate_thoughts(node.content, self.breadth)
                for t in thoughts:
                    score = self._score_thought(t, context)
                    child = ThoughtNode(t, parent=node, score=score)
                    node.add_child(child)
                    self.all_nodes.append(child)
                    next_frontier.append(child)

            next_frontier = self.merge_thoughts(next_frontier)
            for n in next_frontier:
                if n.merged:
                    n.score = min(1.0, n.score + 0.15)

            next_frontier.sort(key=lambda n: n.score, reverse=True)
            frontier = next_frontier[:self.breadth]

        best = max(self.all_nodes, key=lambda n: n.score)
        return {"best_hypothesis": best.content, "best_score": best.score,
                "path": self._trace_path(best), "nodes_explored": len(self.all_nodes),
                "merged_convergent_paths": sum(1 for n in self.all_nodes if n.merged)}


class WorldModel:
    def __init__(self, llm):
        self.llm = llm

    def simulate_action(self, recommendation: str) -> str:
        prompt = (f"Given this automotive repair recommendation: {recommendation}\n"
                  f"Simulate the likely real-world outcome if the user follows it. Be concise.")
        try:
            return self.llm.invoke(prompt).strip()
        except Exception:
            return "Simulation unavailable — proceeding with recommendation as-is."

    def evaluate_recommendation(self, recommendation: str) -> Dict[str, Any]:
        outcome = self.simulate_action(recommendation)
        bad_signals = ["fail", "danger", "worse", "damage", "unsafe", "incorrect"]
        rating = "BAD" if any(s in outcome.lower() for s in bad_signals) else "GOOD"
        return {"recommendation": recommendation, "simulated_outcome": outcome, "rating": rating}

    def validate_and_revise(self, recommendation: str) -> Dict[str, Any]:
        evaluation = self.evaluate_recommendation(recommendation)
        if evaluation["rating"] == "BAD":
            revise_prompt = (f"This recommendation may cause a bad outcome: {recommendation}\n"
                              f"Predicted issue: {evaluation['simulated_outcome']}\n"
                              f"Provide a safer, revised recommendation.")
            try:
                revised = self.llm.invoke(revise_prompt).strip()
            except Exception:
                revised = recommendation + " (Note: recommend professional inspection before proceeding.)"
            evaluation["revised_recommendation"] = revised
            evaluation["was_revised"] = True
        else:
            evaluation["revised_recommendation"] = recommendation
            evaluation["was_revised"] = False
        return evaluation