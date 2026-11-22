import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import verification as V  # noqa: E402
from app.store import Session  # noqa: E402


def _scores(W, x, rows, bases):
    resp = W @ x
    return np.array([bases[i] + resp[rows[i]] for i in range(len(rows))])


# ---------------------------------------------------------------- pure solver


def test_minimal_k_with_a_tied_original_top_word():
    # Original scores: words 0 and 1 are both best (equal).  The ZIP's factors
    # give merged scores 10, 9, 9, 0: word 1 is tied with word 2, so its
    # most-unfavourable merged rank is 3.
    W = np.array([[10.0], [10.0], [0.0], [0.0]])
    B_zip = np.eye(4)
    A_zip = np.array([[10.0], [9.0], [9.0], [0.0]])
    pairs = [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    out = V.evaluate(W, np.array([1.0]), B_zip, A_zip, pairs, 0.0)
    assert out["k"] == 3
    assert out["top_rows"] == [0, 1]
    assert [c["rank"] for c in out["candidates"]] == [1, 3, 3, 4]
    assert out["missed_rows"] == [1]
    assert out["previous_k"] == 2


def test_k_is_one_when_the_single_top_word_wins_outright():
    W = np.array([[5.0], [4.0], [0.0]])
    B_zip = np.eye(3)
    A_zip = np.array([[5.0], [4.0], [0.0]])
    pairs = [(0, 0.0), (1, 0.0), (2, 0.0)]
    out = V.evaluate(W, np.array([1.0]), B_zip, A_zip, pairs, 0.0)
    assert out["k"] == 1
    assert out["top_rows"] == [0]
    assert out["missed_rows"] == []
    assert out["previous_k"] == 0


def test_epsilon_widens_the_original_top_set():
    W = np.array([[5.0], [4.5], [0.0]])
    B_zip = np.eye(3)
    A_zip = np.array([[5.0], [4.5], [0.0]])
    pairs = [(0, 0.0), (1, 0.0), (2, 0.0)]
    assert V.evaluate(W, np.array([1.0]), B_zip, A_zip, pairs, 0.0)["top_rows"] == [0]
    out = V.evaluate(W, np.array([1.0]), B_zip, A_zip, pairs, 1.0)
    assert out["top_rows"] == [0, 1]
    assert out["k"] == 2


def test_merged_ranking_uses_zip_factors_not_w():
    # The ZIP's factors produce a different order from W.  Candidate word 0 is
    # the original top word; the ZIP pushes it down to merged rank 3.
    W = np.array([[5.0], [4.0], [0.0]])
    B_zip = np.eye(3)
    A_zip = np.array([[0.0], [1.0], [2.0]])
    pairs = [(0, 0.0), (1, 0.0), (2, 0.0)]
    out = V.evaluate(W, np.array([1.0]), B_zip, A_zip, pairs, 0.0)
    assert out["top_rows"] == [0]
    assert out["candidates"][0]["merged_score"] == pytest.approx(0.0)
    assert out["candidates"][0]["rank"] == 3
    assert out["k"] == 3
    assert out["missed_rows"] == [0]


def test_base_scores_are_added_per_row():
    W = np.array([[1.0], [2.0], [3.0]])
    B_zip = np.zeros((3, 0))
    A_zip = np.zeros((0, 1))
    pairs = [(0, 100.0), (1, 0.0), (2, 0.0)]
    out = V.evaluate(W, np.array([1.0]), B_zip, A_zip, pairs, 0.0)
    assert out["candidates"][0]["original_score"] == pytest.approx(101.0)
    assert out["top_rows"] == [0]


def test_solver_rejects_bad_inputs():
    W = np.zeros((4, 3))
    B = np.zeros((4, 0))
    A = np.zeros((0, 3))
    good = [(0, 0.0), (1, 0.0)]
    with pytest.raises(V.CandidateError):
        V.evaluate(W, np.zeros(3), B, A, [(0, 0.0), (9, 0.0)], 0.0)
    with pytest.raises(V.CandidateError):
        V.evaluate(W, np.zeros(4), B, A, good, 0.0)  # wrong activation dim
    with pytest.raises(V.CandidateError):
        V.evaluate(W, np.zeros(3), np.zeros((4, 1)), A, good, 0.0)  # dim mismatch
    with pytest.raises(V.CandidateError):
        V.evaluate(W, np.array([np.nan, 0, 0]), B, A, good, 0.0)
    with pytest.raises(V.CandidateError):
        V.evaluate(W, np.zeros(3), B, A, good, -1.0)


def test_parser_enforces_pairing_and_bounds():
    with pytest.raises(V.CandidateError):
        V.parse_candidate_pairs(["0", ""], ["1", "2"])  # missing row
    with pytest.raises(V.CandidateError):
        V.parse_candidate_pairs(["0", "1"], ["1", ""])  # missing base
    with pytest.raises(V.CandidateError):
        V.parse_candidate_pairs(["0"], ["1"])  # fewer than 2
    with pytest.raises(V.CandidateError):
        V.parse_candidate_pairs([str(i) for i in range(9)], ["0"] * 9)  # > 8
    with pytest.raises(V.CandidateError):
        V.parse_candidate_pairs(["0", "0"], ["1", "2"])  # duplicate row
    ok = V.parse_candidate_pairs(["0", "2", "", ""], ["1", "2", "", ""])
    assert ok == [(0, 1.0), (2, 2.0)]
    with pytest.raises(V.CandidateError):
        V.parse_epsilon("nan")
    with pytest.raises(V.CandidateError):
        V.parse_epsilon("-0.5")
    with pytest.raises(V.CandidateError):
        V.parse_activation_index("3", 3)


# --------------------------------------------------------- session currency


def test_upload_invalidates_result_and_inflight_job():
    session = Session(id="s")
    session.verification = object()
    session.verification_job = 5
    session.state_epoch = 2
    session.invalidate_verification("upload")
    assert session.verification is None
    assert session.verification_job == 6
    assert session.state_epoch == 3


def test_repackage_invalidates_result_but_keeps_draft():
    session = Session(id="s")
    session.verification = object()
    session.verification_draft = {"epsilon": 1.0}
    session.invalidate_verification("package")
    assert session.verification is None
    assert session.verification_draft == {"epsilon": 1.0}
    assert session.state_epoch == 1
