"""
pinecone_manager.py — Extended Pinecone Operations
ExtendedPineconeStore wraps LangChain's PineconeVectorStore with production
features: namespaces, native hybrid search, MMR, score thresholds, backup/restore.

Exposes a similarity_search(query, k) method compatible with LangChain's
retriever interface so this object can be passed directly as `vector_store`
into agents and tools without any adapter.
"""

import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


class ExtendedPineconeStore:
    def __init__(self, index_name: str, embeddings, dimension: int = 384):
        self.index_name = index_name
        self.embeddings = embeddings
        self.dimension = dimension
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        if index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name, dimension=dimension, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)
        self._default_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # ── Namespace-aware store access ────────────────────────

    def _store(self, namespace: Optional[str] = None) -> PineconeVectorStore:
        if not namespace:
            return self._default_store
        return PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings, namespace=namespace)

    # ── Core retrieval interface (LangChain-compatible) ─────

    def similarity_search(self, query: str, k: int = 5, namespace: Optional[str] = None,
                           filter: Optional[Dict] = None):
        return self._store(namespace).similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int = 5, namespace: Optional[str] = None,
                                      filter: Optional[Dict] = None):
        return self._store(namespace).similarity_search_with_score(query, k=k, filter=filter)

    def similarity_search_threshold(self, query: str, k: int = 5, score_threshold: float = 0.5,
                                     namespace: Optional[str] = None):
        results = self.similarity_search_with_score(query, k=k, namespace=namespace)
        return [doc for doc, score in results if score >= score_threshold]

    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 20,
                    lambda_mult: float = 0.5, namespace: Optional[str] = None):
        return self._store(namespace).max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    def hybrid_search(self, query: str, k: int = 5, namespace: Optional[str] = None):
        """Native dense + sparse (BM25) hybrid search, server-side, via pinecone-text."""
        try:
            from pinecone_text.sparse import BM25Encoder
            encoder = BM25Encoder().default()
            sparse_vec = encoder.encode_queries(query)
            dense_vec = self.embeddings.embed_query(query)
            results = self.index.query(vector=dense_vec, sparse_vector=sparse_vec, top_k=k,
                namespace=namespace, include_metadata=True)
            return results.get("matches", [])
        except ImportError:
            return self.similarity_search(query, k=k, namespace=namespace)

    # ── Document management ─────────────────────────────────

    def add_documents(self, docs: List, namespace: Optional[str] = None):
        return self._store(namespace).add_documents(docs)

    def delete_by_source(self, filename: str, namespace: Optional[str] = None) -> bool:
        try:
            self.index.delete(filter={"source": {"$eq": filename}}, namespace=namespace)
            return True
        except Exception:
            return False

    def fetch_by_id(self, ids: List[str], namespace: Optional[str] = None):
        return self.index.fetch(ids=ids, namespace=namespace)

    # ── Namespace management ────────────────────────────────

    def list_namespaces(self) -> List[str]:
        stats = self.index.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())

    def delete_namespace(self, namespace: str):
        return self.index.delete(delete_all=True, namespace=namespace)

    # ── Stats + health ───────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", self.dimension),
            "namespaces": stats.get("namespaces", {}),
            "index_fullness": stats.get("index_fullness", 0),
            "index_name": self.index_name,
        }

    # ── Backup / restore via Pinecone collections ───────────

    def create_backup(self, collection_name: str) -> bool:
        try:
            self.pc.create_collection(name=collection_name, source=self.index_name)
            return True
        except Exception:
            return False

    def restore_from_backup(self, collection_name: str, new_index_name: str) -> bool:
        try:
            self.pc.create_index_for_model(name=new_index_name, source_collection=collection_name)
            return True
        except Exception:
            return False


# ── Module-level convenience function (used directly by main.py) ──

def list_namespaces(index_name: Optional[str] = None) -> List[str]:
    index_name = index_name or os.environ.get("PINECONE_INDEX", "autodiag")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    return list(stats.get("namespaces", {}).keys())