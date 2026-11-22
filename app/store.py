"""In-memory session state.

A session tracks the single linear layer being merged.  Only one published
package exists per session at a time:

* a merge only replaces the published package **after** the new package has
  been built and its exported factors have passed a self-test;
* a failed merge (for example, an infeasible rank limit) leaves the published
  package and its manifest metadata untouched.

Session state is process-local.  The container runs one worker, which is
adequate for this single-user deployment tool.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PublishedPackage:
    data: bytes
    manifest: dict
    generation: int
    rank_limit: int
    protected_indices: list[int]
    input_fingerprint: str


@dataclass
class Session:
    id: str
    B1: Optional[np.ndarray] = None
    A1: Optional[np.ndarray] = None
    B2: Optional[np.ndarray] = None
    A2: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    rank_limit: int = 8
    protected_indices: list[int] = field(default_factory=list)
    published: Optional[PublishedPackage] = None
    generation: int = 0
    last_error: Optional[dict] = None
    last_notice: Optional[dict] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def adapters_ready(self) -> bool:
        return all(m is not None for m in (self.B1, self.A1, self.B2, self.A2))

    def activations_ready(self) -> bool:
        return self.X is not None

    def inputs_ready(self) -> bool:
        return self.adapters_ready() and self.activations_ready()

    def layer_dims(self):
        if not self.adapters_ready():
            return None
        return int(self.A1.shape[1]), int(self.B1.shape[0])

    def input_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in (
            ("B1", self.B1),
            ("A1", self.A1),
            ("B2", self.B2),
            ("A2", self.A2),
            ("X", self.X),
        ):
            digest.update(name.encode())
            if value is not None:
                digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
            else:
                digest.update(b"-")
        return digest.hexdigest()

    def package_is_stale(self) -> bool:
        return (
            self.published is not None
            and self.published.input_fingerprint != self.input_fingerprint()
        )

    def summary(self) -> dict:
        dims = self.layer_dims()
        return {
            "session_id": self.id,
            "adapters_ready": self.adapters_ready(),
            "activations_ready": self.activations_ready(),
            "layer_input_dim": dims[0] if dims else None,
            "layer_output_dim": dims[1] if dims else None,
            "activation_count": int(self.X.shape[1]) if self.X is not None else 0,
            "adapter_ranks": (
                [int(self.B1.shape[1]), int(self.B2.shape[1])]
                if self.adapters_ready()
                else None
            ),
            "rank_limit": int(self.rank_limit),
            "protected_indices": list(self.protected_indices),
            "published": None
            if self.published is None
            else {
                "generation": self.published.generation,
                "rank_limit": self.published.rank_limit,
                "protected_indices": list(self.published.protected_indices),
                "shared_dim": self.published.manifest.get("shared_dim"),
                "created_at": self.published.manifest.get("created_at"),
                "size_bytes": len(self.published.data),
                "stale": self.package_is_stale(),
            },
            "last_error": self.last_error,
            "last_notice": self.last_notice,
        }


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        session = Session(id=uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)
