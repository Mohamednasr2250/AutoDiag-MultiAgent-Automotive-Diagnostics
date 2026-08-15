"""
chunk_graph.py — Chunk Similarity Graph for RAG
Pure-Python term-overlap cosine similarity graph. No embedding model required.
Graph-based retrieval (chunk + neighbors) and dedup via connected components.
"""

import re
from collections import Counter
from typing import List, Dict, Set, Any

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document


def _vectorize(text: str) -> Counter:
    words = re.findall(r"[a-z]+", text.lower())
    return Counter(words)


def _cosine_sim(v1: Counter, v2: Counter) -> float:
    common = set(v1) & set(v2)
    dot = sum(v1[w] * v2[w] for w in common)
    norm1 = sum(v * v for v in v1.values()) ** 0.5
    norm2 = sum(v * v for v in v2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class ChunkGraph:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.chunks: List[Document] = []
        self._vectors: List[Counter] = []
        self.adjacency: Dict[int, Set[int]] = {}

    def add_chunks(self, chunks: List[Document]):
        start = len(self.chunks)
        for i, doc in enumerate(chunks):
            idx = start + i
            self.chunks.append(doc)
            self.adjacency[idx] = set()
            self._vectors.append(_vectorize(doc.page_content))

        n = len(self.chunks)
        for i in range(start, n):
            for j in range(i):
                sim = _cosine_sim(self._vectors[i], self._vectors[j])
                if sim >= self.threshold:
                    self.adjacency[i].add(j)
                    self.adjacency[j].add(i)

    def search_with_graph(self, query: str, top_k: int = 2, hops: int = 1) -> List[Document]:
        if not self.chunks:
            return []
        qvec = _vectorize(query)
        scored = [(_cosine_sim(qvec, self._vectors[i]), i) for i in range(len(self.chunks))]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [i for score, i in scored[:top_k] if score > 0] or [scored[0][1]]

        result_idx = set(top)
        frontier = set(top)
        for _ in range(hops):
            next_frontier = set()
            for idx in frontier:
                next_frontier |= self.adjacency.get(idx, set())
            result_idx |= next_frontier
            frontier = next_frontier

        return [self.chunks[i] for i in result_idx]

    def get_connected_components(self) -> List[Set[int]]:
        visited: Set[int] = set()
        components = []
        for i in range(len(self.chunks)):
            if i in visited:
                continue
            stack = [i]
            comp = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                stack.extend(self.adjacency.get(node, set()) - visited)
            components.append(comp)
        return components

    def deduplicate(self) -> List[Document]:
        components = self.get_connected_components()
        representatives = []
        for comp in components:
            rep_idx = max(comp, key=lambda i: len(self.chunks[i].page_content))
            representatives.append(self.chunks[rep_idx])
        return representatives

    def get_stats(self) -> Dict[str, Any]:
        total_edges = sum(len(v) for v in self.adjacency.values()) // 2
        return {
            "total_chunks": len(self.chunks), "total_edges": total_edges,
            "components": len(self.get_connected_components()), "threshold": self.threshold
        }


# ── Module-level global graph + convenience functions ───────
# main.py calls add_chunks_to_graph(docs) on every manual upload and
# get_graph_stats() from the /graph/chunks/stats endpoint — both operate
# on one process-wide ChunkGraph instance rather than requiring callers
# to manage a ChunkGraph object themselves.

_global_chunk_graph = ChunkGraph(threshold=0.3)


def add_chunks_to_graph(chunks: List[Document]) -> Dict[str, Any]:
    _global_chunk_graph.add_chunks(chunks)
    return _global_chunk_graph.get_stats()


def get_graph_stats() -> Dict[str, Any]:
    return _global_chunk_graph.get_stats()


def search_chunk_graph(query: str, top_k: int = 3, hops: int = 1) -> List[Document]:
    return _global_chunk_graph.search_with_graph(query, top_k=top_k, hops=hops)


def deduplicate_chunk_graph() -> List[Document]:
    return _global_chunk_graph.deduplicate()