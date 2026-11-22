import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import factorization as F  # noqa: E402


def random_problem(seed, in_dim=None, out_dim=None, r1=None, r2=None, n_act=None):
    rng = np.random.default_rng(seed)
    in_dim = int(in_dim or rng.integers(2, 17))
    out_dim = int(out_dim or rng.integers(2, 17))
    r1 = int(r1 or rng.integers(1, 7))
    r2 = int(r2 or rng.integers(1, 7))
    n_act = int(n_act or rng.integers(1, 9))
    B1 = rng.normal(size=(out_dim, r1))
    A1 = rng.normal(size=(r1, in_dim))
    B2 = rng.normal(size=(out_dim, r2))
    A2 = rng.normal(size=(r2, in_dim))
    X = rng.normal(size=(in_dim, n_act))
    return B1, A1, B2, A2, X


@pytest.mark.parametrize("seed", range(30))
def test_protected_response_is_exact(seed):
    B1, A1, B2, A2, X = random_problem(seed)
    W = B1 @ A1 + B2 @ A2
    prot = sorted(
        np.random.default_rng(seed + 100).choice(
            X.shape[1], size=max(1, X.shape[1] // 2), replace=False
        ).tolist()
    )
    Xs = X[:, prot]
    k = F.minimum_feasible_rank(W, Xs)
    for rank_limit in (k, k + 1, k + 3, k + 10):
        B, A = F.factorize_update(W, Xs, rank_limit)
        assert B.shape[1] == A.shape[0] <= rank_limit
        err = np.abs(B @ (A @ Xs) - W @ Xs)
        assert err.max() <= 1e-12 * max(1.0, np.linalg.norm(W) * np.linalg.norm(Xs))


@pytest.mark.parametrize("seed", range(20))
def test_frobenius_distance_matches_theoretical_optimum(seed):
    B1, A1, B2, A2, X = random_problem(seed)
    W = B1 @ A1 + B2 @ A2
    Xs = X[:, : max(1, X.shape[1] // 2)]
    k = F.minimum_feasible_rank(W, Xs)
    for rank_limit in range(k, k + 4):
        B, A = F.factorize_update(W, Xs, rank_limit)
        assert F.frobenius_distance(B, A, W) == pytest.approx(
            F.theoretical_min_distance(W, Xs, rank_limit), rel=1e-9, abs=1e-9
        )


def test_below_minimum_rank_is_infeasible():
    B1, A1, B2, A2, X = random_problem(5)
    W = B1 @ A1 + B2 @ A2
    Xs = X[:, :3]
    k = F.minimum_feasible_rank(W, Xs)
    assert k >= 1
    with pytest.raises(F.InfeasibleRankError) as excinfo:
        F.factorize_update(W, Xs, k - 1)
    assert excinfo.value.min_rank == k
    assert excinfo.value.requested_rank == k - 1


def test_minimum_rank_bound_is_attained():
    """rank(W X) is a lower bound and the construction attains it."""
    B1, A1, B2, A2, X = random_problem(9, in_dim=10, out_dim=8, r1=3, r2=4, n_act=5)
    W = B1 @ A1 + B2 @ A2
    Xs = X[:, [0, 2, 4]]
    k = F.minimum_feasible_rank(W, Xs)
    B, A = F.factorize_update(W, Xs, k)
    assert B.shape[1] <= k
    assert np.allclose(B @ (A @ Xs), W @ Xs, atol=1e-11)


def test_full_rank_request_reproduces_update_exactly():
    # X spans the whole input space, so the constraint forces BA X = W X to
    # cover every direction: the result must be exactly W.
    B1, A1, B2, A2, _ = random_problem(13, in_dim=9, out_dim=7)
    W = B1 @ A1 + B2 @ A2
    X = np.eye(9)
    B, A = F.factorize_update(W, X, 32)
    assert np.allclose(B @ A, W, atol=1e-10)
    assert F.frobenius_distance(B, A, W) < 1e-10


def test_full_rank_with_partial_protection_is_optimal_not_exact():
    # When X does not span the input space the constraint only fixes one
    # direction; even a generous rank limit leaves a nonzero optimum.
    B1, A1, B2, A2, X = random_problem(14, in_dim=9, out_dim=7)
    W = B1 @ A1 + B2 @ A2
    B, A = F.factorize_update(W, X[:, [0]], 32)
    assert np.abs(B @ (A @ X[:, [0]]) - W @ X[:, [0]]).max() < 1e-10
    assert F.frobenius_distance(B, A, W) == pytest.approx(
        F.theoretical_min_distance(W, X[:, [0]], 32), rel=1e-9, abs=1e-9
    )


def test_zero_protected_response():
    B1 = np.zeros((4, 2))
    A1 = np.zeros((2, 5))
    B2 = np.zeros((4, 2))
    A2 = np.zeros((2, 5))
    W = B1 @ A1 + B2 @ A2
    X = np.ones((5, 2))
    B, A = F.factorize_update(W, X, 3)
    assert B.shape[1] == 0 and A.shape[0] == 0
    assert F.response_max_abs_error(B, A, W, X) == 0.0


def test_dimension_limits_enforced():
    with pytest.raises(F.ShapeError):
        F.validate_adapters(
            np.zeros((33, 1)), np.zeros((1, 4)), np.zeros((33, 1)), np.zeros((1, 4))
        )
    with pytest.raises(F.ShapeError):
        F.validate_adapters(
            np.zeros((4, 1)), np.zeros((1, 33)), np.zeros((4, 1)), np.zeros((1, 33))
        )


def test_mismatched_layer_rejected():
    with pytest.raises(F.ShapeError):
        F.validate_adapters(
            np.zeros((4, 2)), np.zeros((2, 5)), np.zeros((5, 2)), np.zeros((2, 5))
        )


def test_non_finite_rejected():
    B1 = np.zeros((4, 2))
    A1 = np.zeros((2, 5))
    A1[0, 0] = np.nan
    with pytest.raises(F.DataError):
        F.validate_adapters(B1, A1, np.zeros((4, 2)), np.zeros((2, 5)))


def test_merge_reports_metadata_and_self_check():
    B1, A1, B2, A2, X = random_problem(21)
    result = F.merge(B1, A1, B2, A2, X, 6, [0, 1])
    assert result["rank_limit"] == 6
    assert result["shared_dim"] <= 6
    assert result["protected_indices"] == [0, 1]
    assert result["response_max_abs_error"] <= 1e-9
    assert result["minimum_feasible_rank"] == F.minimum_feasible_rank(result["W"], X[:, [0, 1]])


def test_merge_requires_protected_selection():
    B1, A1, B2, A2, X = random_problem(22)
    with pytest.raises(F.MergeError):
        F.merge(B1, A1, B2, A2, X, 4, [])


def test_merge_rejects_bad_indices():
    B1, A1, B2, A2, X = random_problem(23)
    with pytest.raises(F.MergeError):
        F.merge(B1, A1, B2, A2, X, 4, [0, X.shape[1]])
    with pytest.raises(F.MergeError):
        F.merge(B1, A1, B2, A2, X, 4, [-1])
