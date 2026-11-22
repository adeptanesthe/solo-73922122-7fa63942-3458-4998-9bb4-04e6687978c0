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

    @property
    def sha256(self) -> str:
        """Content hash of the downloadable ZIP.

        Candidate verification is bound to this hash, so a different package
        that happens to reuse an in-memory generation can never be accepted.
        """
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class VerificationResult:
    """A candidate-count verification bound to one package generation."""

    generation: int
    package_sha256: str
    input_fingerprint: str
    activation_index: int
    epsilon: float
    pairs: list[tuple[int, float]]
    outcome: dict
    created_at: str


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
    # Candidate-count verification.  ``state_epoch`` changes whenever the
    # loaded inputs or the published package change; an in-flight solve that
    # started under an older epoch is discarded instead of being displayed.
    verification: Optional[VerificationResult] = None
    verification_error: Optional[dict] = None
    verification_draft: Optional[dict] = None
    verification_running: Optional[int] = None
    verification_job: int = 0
    state_epoch: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def adapters_ready(self) -> bool:
        return all(m is not None for m in (self.B1, self.A1, self.B2, self.A2))

    def activations_ready(self) -> bool:
        return self.X is not None

    def inputs_ready(self) -> bool:
        return self.adapters_ready() and self.activations_ready()

    def combined_update(self) -> Optional[np.ndarray]:
        if not self.adapters_ready():
            return None
        return self.B1 @ self.A1 + self.B2 @ self.A2

    def invalidate_verification(self, reason: str) -> None:
        """Withdraw any stored result and discard in-flight solves.

        Incrementing ``state_epoch`` is what makes a solve that is still
        running fail its final currency check.
        """
        self.state_epoch += 1
        self.verification = None
        self.verification_error = None
        self.verification_running = None
        self.verification_job += 1
        if reason == "upload":
            self.verification_draft = None

    def begin_verification_attempt(self) -> int:
        """Supersede the previous attempt and return the new job id.

        The old conclusion described a different request (different activation,
        epsilon or candidate set), so it is withdrawn rather than left next to
        the new one.  Any still-running solve also loses its job id and will be
        discarded when it finishes.
        """
        self.verification_job += 1
        self.verification_running = None
        self.verification = None
        self.verification_error = None
        return self.verification_job

    def verification_is_current(self) -> bool:
        result = self.verification
        if result is None or self.published is None:
            return False
        if self.package_is_stale():
            return False
        return (
            result.generation == self.published.generation
            and result.package_sha256 == self.published.sha256
            and result.input_fingerprint == self.input_fingerprint()
            and result.input_fingerprint == self.published.input_fingerprint
        )

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

    def verification_summary(self) -> Optional[dict]:
        if self.verification is None:
            return None
        return {
            "generation": self.verification.generation,
            "package_sha256": self.verification.package_sha256,
            "input_fingerprint": self.verification.input_fingerprint,
            "activation_index": self.verification.activation_index,
            "epsilon": self.verification.epsilon,
            "candidate_count": self.verification.outcome["candidate_count"],
            "k": self.verification.outcome["k"],
            "previous_k": self.verification.outcome["previous_k"],
            "created_at": self.verification.created_at,
            "current": self.verification_is_current(),
            "outcome": self.verification.outcome,
        }

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
            "verification": self.verification_summary(),
            "verification_error": self.verification_error,
            "verification_draft": self.verification_draft,
            "verification_running": self.verification_running is not None,
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
