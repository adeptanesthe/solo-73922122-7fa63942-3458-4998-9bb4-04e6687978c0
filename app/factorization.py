"""Constrained low-rank merging of two LoRA adapters for one linear layer.

The original update is ``W = B1 @ A1 + B2 @ A2``.  A single adapter ``(B, A)``
with shared inner dimension ``<= rank_limit`` must reproduce ``W`` exactly on a
selected set of activations ``X``::

    B @ (A @ X) == W @ X

and, among all factorisations satisfying that hard constraint, minimise
``||B @ A - W||_F``.

Optimal construction
--------------------
Let ``P`` be an orthonormal basis of ``span(X)`` and ``U1`` an orthonormal
basis of ``col(W P)`` (dimension ``k = rank(W X)``).

* The constraint is equivalent to ``(B A) P = W P``.  It fixes the action of
  ``BA`` on ``span(X)`` and leaves ``span(X)^perp`` free.
* Any feasible ``BA`` has rank at least ``k`` (because
  ``rank(M P) <= rank(M)``), so ``rho = rank_limit - k`` extra dimensions
  remain.  Writing ``M = M P P^T + M (I - P P^T)``, the Frobenius objective
  splits orthogonally into a forced part and a free part.  Choosing the
  ``U1``-row block of the free part optimally gives ``U1 U1^T W``; the rest is
  a rank-``rho`` approximation of
  ``R = (I - U1 U1^T) W (I - P P^T)``, whose best solution is the truncated
  SVD ``U2 S2 V2^T`` (Eckart-Young).  Hence::

      M = U1 U1^T W + U2 U2^T W (I - P P^T)

  with ``M P = W P`` and ``rank(M) <= k + rho = rank_limit``.
* A factorisation is ``B = [U1 | U2]`` and
  ``A = [U1^T W ; U2^T W (I - P P^T)]``.

If ``rank_limit < k`` the constraint is infeasible; ``k`` is the minimum
feasible rank and is reported to the caller.
"""

from __future__ import annotations

import numpy as np

# Singular values below RTOL * sigma_max are treated as numerical zero.
RANK_RTOL = 1e-11
# Relative tolerance for the internal response-preservation assertion.
RESPONSE_RTOL = 1e-9
MAX_LAYER_DIM = 32
MAX_ADAPTER_RANK = 1024
MAX_ACTIVATION_VECTORS = 2048


class MergeError(ValueError):
    """Merge error with a message safe to show to the user."""


class ShapeError(MergeError):
    pass


class DataError(MergeError):
    pass


class InfeasibleRankError(MergeError):
    def __init__(self, requested_rank: int, min_rank: int):
        self.requested_rank = int(requested_rank)
        self.min_rank = int(min_rank)
        super().__init__(
            f"秩上限 {requested_rank} 过小：在当前保护项下最低可行秩为 {min_rank}"
        )


def _matrix(name: str, value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{name} 不是数值矩阵") from exc
    if arr.ndim != 2:
        raise ShapeError(f"{name} 必须是二维矩阵，实际为 {arr.ndim} 维")
    if arr.size and not np.all(np.isfinite(arr)):
        raise DataError(f"{name} 含有 NaN 或无穷值")
    return arr


def validate_adapters(B1, A1, B2, A2):
    """Validate the two adapters and return them as float64 matrices."""
    B1 = _matrix("适配器 1 的 B", B1)
    A1 = _matrix("适配器 1 的 A", A1)
    B2 = _matrix("适配器 2 的 B", B2)
    A2 = _matrix("适配器 2 的 A", A2)

    for tag, B, A in (("1", B1, A1), ("2", B2, A2)):
        if B.shape[1] != A.shape[0]:
            raise ShapeError(
                f"适配器 {tag} 的因子不匹配：B 有 {B.shape[1]} 个秩列，"
                f"A 有 {A.shape[0]} 个秩行（约定 B 为 out×r、A 为 r×in）"
            )
        if not (1 <= B.shape[1] <= MAX_ADAPTER_RANK):
            raise ShapeError(
                f"适配器 {tag} 的秩必须位于 1..{MAX_ADAPTER_RANK}，实际为 {B.shape[1]}"
            )

    if B1.shape[0] != B2.shape[0] or A1.shape[1] != A2.shape[1]:
        raise ShapeError(
            "两份适配器目标层不一致："
            f"适配器 1 作用于 {A1.shape[1]}→{B1.shape[0]}，"
            f"适配器 2 作用于 {A2.shape[1]}→{B2.shape[0]}；"
            "两个因子需分别满足 B 为 out×r、A 为 r×in"
        )

    out_dim, in_dim = B1.shape[0], A1.shape[1]
    if not (1 <= in_dim <= MAX_LAYER_DIM) or not (1 <= out_dim <= MAX_LAYER_DIM):
        raise ShapeError(
            f"层的输入/输出维度必须各在 1..{MAX_LAYER_DIM} 之间，"
            f"实际为 input={in_dim}, output={out_dim}"
        )
    return B1, A1, B2, A2


def validate_activations(X, in_dim: int) -> np.ndarray:
    X = _matrix("激活矩阵 X", X)
    if X.shape[0] != in_dim:
        raise ShapeError(
            f"激活矩阵的行数必须等于层输入维度 {in_dim}，实际为 {X.shape[0]}；"
            "约定每个激活向量为一列"
        )
    if X.shape[1] == 0:
        raise ShapeError("激活矩阵至少需要一列激活向量")
    if X.shape[1] > MAX_ACTIVATION_VECTORS:
        raise ShapeError(
            f"激活向量过多（最多 {MAX_ACTIVATION_VECTORS} 列），实际为 {X.shape[1]}"
        )
    return X


def combine_adapters(B1, A1, B2, A2) -> np.ndarray:
    B1, A1, B2, A2 = validate_adapters(B1, A1, B2, A2)
    return B1 @ A1 + B2 @ A2


def matrix_rank(M: np.ndarray, rtol: float = RANK_RTOL) -> int:
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int(np.count_nonzero(s > rtol * s[0]))


def _orthonormal_basis(M: np.ndarray, rtol: float = RANK_RTOL) -> np.ndarray:
    """Orthonormal basis (columns) of the column space of ``M``."""
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((M.shape[0], 0))
    return U[:, : int(np.count_nonzero(s > rtol * s[0]))]


def minimum_feasible_rank(W: np.ndarray, X_selected: np.ndarray) -> int:
    """Smallest shared dimension able to reproduce ``W @ X_selected``.

    ``rank(B A X) <= rank(B A)`` and ``B A X = W X``, hence
    ``rank(B A) >= rank(W X)``.  The bound is attained by the construction in
    :func:`factorize_update`.
    """
    if X_selected.size == 0:
        return 0
    return matrix_rank(W @ X_selected)


def _k_and_bases(W, X_selected):
    P = _orthonormal_basis(X_selected)
    WX = W @ X_selected
    U1 = _orthonormal_basis(WX)
    return P, U1, matrix_rank(WX)


def theoretical_min_distance(W, X_selected, rank_limit: int) -> float:
    """Minimum achievable ``||BA - W||_F`` for the constrained problem.

    Useful as an independent check of :func:`factorize_update`.
    """
    P, U1, k = _k_and_bases(W, X_selected)
    if rank_limit < k:
        return float("nan")
    rho = int(rank_limit) - k
    residual = W - U1 @ (U1.T @ W)
    if P.shape[1]:
        residual = residual @ (np.eye(W.shape[1]) - P @ P.T)
    if rho <= 0 or residual.size == 0:
        return float(np.linalg.norm(residual, "fro"))
    s = np.linalg.svd(residual, compute_uv=False)
    if not s.size or s[0] == 0.0:
        return 0.0
    usable = int(np.count_nonzero(s > RANK_RTOL * s[0]))
    keep = min(rho, usable)
    return float(np.sqrt(np.sum(s[keep:] ** 2)))


def factorize_update(W: np.ndarray, X_selected: np.ndarray, rank_limit: int):
    """Return ``(B, A)``: optimal constrained rank-``rank_limit`` merge.

    Raises :class:`InfeasibleRankError` when ``rank_limit`` is below the rank
    of the protected response.
    """
    if not isinstance(rank_limit, (int, np.integer)) or isinstance(rank_limit, bool):
        raise DataError("秩上限必须是整数")
    if rank_limit < 0:
        raise DataError("秩上限不能为负数")

    P, U1, k = _k_and_bases(W, X_selected)
    if rank_limit < k:
        raise InfeasibleRankError(int(rank_limit), k)

    if k == 0:
        # No protected response (W X = 0).  The constraint (BA) P = 0 is
        # automatically satisfied by the truncated SVD of W because W P = 0.
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        usable = int(np.count_nonzero(s > RANK_RTOL * s[0])) if s.size and s[0] > 0 else 0
        keep = min(int(rank_limit), usable)
        return U[:, :keep] * np.sqrt(s[:keep]), np.sqrt(s[:keep])[:, None] * Vt[:keep, :]

    extra = int(rank_limit) - k
    # Free part: the component of W outside col(U1) and outside span(X).
    residual = W - U1 @ (U1.T @ W)
    if P.shape[1]:
        residual = residual @ (np.eye(W.shape[1]) - P @ P.T)

    U2 = np.zeros((W.shape[0], 0))
    if extra > 0 and residual.size:
        U2_full, s2, _ = np.linalg.svd(residual, full_matrices=False)
        usable = (
            int(np.count_nonzero(s2 > RANK_RTOL * s2[0]))
            if s2.size and s2[0] > 0
            else 0
        )
        U2 = U2_full[:, : min(extra, usable)]

    B = np.hstack([U1, U2]) if U2.shape[1] else U1
    if U2.shape[1]:
        A = np.vstack([U1.T @ W, U2.T @ residual])
    else:
        A = U1.T @ W
    return B, A


def response_max_abs_error(B, A, W, X_selected) -> float:
    target = W @ X_selected
    merged = B @ (A @ X_selected) if B.shape[1] else np.zeros_like(target)
    if target.size == 0:
        return 0.0
    return float(np.max(np.abs(merged - target)))


def frobenius_distance(B, A, W) -> float:
    merged = B @ A if B.shape[1] else np.zeros_like(W)
    return float(np.linalg.norm(merged - W, "fro"))


def merge(B1, A1, B2, A2, X, rank_limit: int, protected_indices):
    """Validate, merge and return the result bundle used by the API."""
    B1, A1, B2, A2 = validate_adapters(B1, A1, B2, A2)
    in_dim = A1.shape[1]
    X = validate_activations(X, in_dim)

    try:
        indices = sorted({int(i) for i in protected_indices})
    except (TypeError, ValueError) as exc:
        raise DataError("保护项下标必须是整数") from exc
    if not indices:
        raise MergeError("至少需要勾选一个必须保住响应的激活向量")
    for idx in indices:
        if idx < 0 or idx >= X.shape[1]:
            raise MergeError(f"保护项下标 {idx} 超出范围（共 {X.shape[1]} 个激活向量）")

    W = B1 @ A1 + B2 @ A2
    X_sel = X[:, indices]

    B, A = factorize_update(W, X_sel, rank_limit)

    shared_dim = int(B.shape[1])
    if shared_dim > int(rank_limit):
        raise MergeError("内部错误：合并后共享维度超过秩上限")

    resp_err = response_max_abs_error(B, A, W, X_sel)
    scale = max(
        1.0,
        float(np.linalg.norm(W, "fro")) * float(np.linalg.norm(X_sel, "fro")),
    )
    if resp_err > RESPONSE_RTOL * scale:
        raise MergeError(f"内部校验失败：所选激活上的响应偏差 {resp_err:.3e} 超过容差")

    frob = frobenius_distance(B, A, W)
    return {
        "B": B,
        "A": A,
        "W": W,
        "rank_limit": int(rank_limit),
        "shared_dim": shared_dim,
        "protected_indices": indices,
        "response_max_abs_error": resp_err,
        "frobenius_distance": frob,
        "frobenius_relative": frob / max(float(np.linalg.norm(W, "fro")), 1e-300),
        "original_update_norm": float(np.linalg.norm(W, "fro")),
        "minimum_feasible_rank": minimum_feasible_rank(W, X_sel),
        "theoretical_min_distance": theoretical_min_distance(W, X_sel, rank_limit),
        "trainable_parameters": int(B.size + A.size),
        "layer_in_dim": int(in_dim),
        "layer_out_dim": int(B1.shape[0]),
    }
