"""
FaissStore: manages the face embedding index.
Uses IndexFlatIP (inner product) on L2-normalized vectors = cosine similarity.
Thread-safe. Persists index + id_map to disk after every write.
"""
from __future__ import annotations
import os
import pickle
import threading
from typing import Optional, List, Tuple

import numpy as np

from core.config import settings

_store_instance: Optional["FaissStore"] = None
_store_lock = threading.Lock()

EMBEDDING_DIM = 512


class FaissStore:
    def __init__(self, index_path: str, map_path: str, dim: int = EMBEDDING_DIM):
        import faiss
        self._faiss = faiss
        self.index_path = index_path
        self.map_path = map_path
        self.dim = dim
        self._index: "faiss.IndexFlatIP" = None
        self._id_map: dict[int, int] = {}  # faiss_row_idx → employee_id
        self._lock = threading.RLock()
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            self._index = self._faiss.read_index(self.index_path)
            with open(self.map_path, "rb") as f:
                self._id_map = pickle.load(f)
            print(f"[FaissStore] Loaded index with {self._index.ntotal} embeddings.")
        else:
            self._index = self._faiss.IndexFlatIP(self.dim)
            self._id_map = {}
            print("[FaissStore] Created new empty index.")

    def add(self, employee_id: int, embedding: np.ndarray):
        """Add an employee's face embedding to the index."""
        with self._lock:
            vec = self._normalize(embedding)
            row_id = self._index.ntotal
            self._index.add(vec)
            self._id_map[row_id] = employee_id
            self._persist()

    def search(
        self, embedding: np.ndarray, top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Find the closest employees to this embedding.
        Returns [(employee_id, cosine_similarity), ...] sorted by similarity descending.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []
            vec = self._normalize(embedding)
            k = min(top_k, self._index.ntotal)
            distances, indices = self._index.search(vec, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx in self._id_map:
                    results.append((self._id_map[idx], float(dist)))
            return results

    def remove_employee(self, employee_id: int):
        """
        Remove all embeddings for an employee.
        Requires full index rebuild (Faiss FlatIP does not support deletion).
        """
        with self._lock:
            rows_to_keep = [
                r for r, eid in self._id_map.items() if eid != employee_id
            ]
            if not rows_to_keep:
                self._index = self._faiss.IndexFlatIP(self.dim)
                self._id_map = {}
            else:
                # Extract all vectors, keep only the ones we want
                all_vecs = np.zeros(
                    (self._index.ntotal, self.dim), dtype=np.float32
                )
                self._faiss.rev_swig_ptr(
                    self._index.get_xb(), self._index.ntotal * self.dim
                ).reshape(-1)
                # Use reconstruct_n for safer extraction
                for i in range(self._index.ntotal):
                    all_vecs[i] = self._index.reconstruct(i)
                kept_vecs = all_vecs[rows_to_keep]
                self._index = self._faiss.IndexFlatIP(self.dim)
                self._index.add(kept_vecs)
                self._id_map = {
                    new_r: self._id_map[old_r]
                    for new_r, old_r in enumerate(rows_to_keep)
                }
            self._persist()

    def count(self) -> int:
        with self._lock:
            return self._index.ntotal

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        vec = embedding.astype(np.float32).reshape(1, self.dim)
        self._faiss.normalize_L2(vec)
        return vec

    def _persist(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self._faiss.write_index(self._index, self.index_path)
        with open(self.map_path, "wb") as f:
            pickle.dump(self._id_map, f)


def get_faiss_store() -> FaissStore:
    """Return singleton FaissStore."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = FaissStore(
                    settings.faiss_index_path,
                    settings.faiss_map_path,
                )
    return _store_instance
