import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


def _npy(arr):
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


@pytest.fixture()
def problem():
    rng = np.random.default_rng(123)
    B1 = rng.normal(size=(6, 3))
    A1 = rng.normal(size=(3, 5))
    B2 = rng.normal(size=(6, 2))
    A2 = rng.normal(size=(2, 5))
    X = rng.normal(size=(5, 8))
    return B1, A1, B2, A2, X


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, problem, X=None):
    B1, A1, B2, A2, X_default = problem
    X = X_default if X is None else X
    return client.post(
        "/upload",
        files={
            "adapter1_bundle": ("a1.npz", _npz(lora_B=B1, lora_A=A1), "application/zip"),
            "adapter2_b": ("b2.npy", _npy(B2), "application/octet-stream"),
            "adapter2_a": ("a2.npy", _npy(A2), "application/octet-stream"),
            "activations": ("x.npy", _npy(X), "application/octet-stream"),
        },
        follow_redirects=False,
    )


def _merge(client, rank, protected):
    return client.post(
        "/merge",
        data={"rank_limit": str(rank), "protected": [str(i) for i in protected]},
        follow_redirects=False,
    )


def _verify(client, activation, epsilon, pairs):
    data = {"activation_index": str(activation), "epsilon": str(epsilon)}
    data["candidate_row"] = [str(row) for row, _ in pairs]
    data["candidate_base"] = [repr(base) for _, base in pairs]
    return client.post("/verify", data=data, follow_redirects=False)


def _wait_for_verification(client, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        page = client.get("/").text
        if "正在用当前这版包求解" not in page:
            return page
        time.sleep(0.02)
    raise AssertionError("verification did not finish in time")


def _zip_factors(client):
    response = client.get("/download")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        B = np.load(io.BytesIO(zf.read("B.npy")))
        A = np.load(io.BytesIO(zf.read("A.npy")))
    return B, A


def test_verify_requires_a_package(client, problem):
    _upload(client, problem)
    _verify(client, 0, 0, [(0, 0.0), (1, 0.0)])
    page = client.get("/").text
    assert "还没有已发布的下载包" in page


def test_verify_without_inputs_is_refused(client):
    _verify(client, 0, 0, [(0, 0.0), (1, 0.0)])
    assert "请先上传两份适配器与激活向量" in client.get("/").text


def test_protected_activation_tie_gives_k_equal_to_tied_words(client, problem):
    """All three candidate scores are equal on a protected activation.

    With an epsilon that absorbs floating-point noise all three words are
    original top words and every merged rank is the worst possible (3), so
    k = 3 and k - 1 = 2 loses every one of them.
    """
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0, 1, 2])

    x = X[:, 0]
    resp = W @ x
    pairs = [(row, -float(resp[row])) for row in (0, 2, 4)]
    assert _verify(client, 0, 1e-6, pairs).status_code == 303
    page = _wait_for_verification(client)

    assert "k = 3" in page
    assert "k − 1 = 2" in page
    assert "激活 #0" in page
    assert "输出行 0、2、4" in page
    assert "第 1 版" in page
    assert "候选数核定" in page


def test_protected_activation_clear_winner_gives_k_one(client, problem):
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0, 1, 2])

    resp = W @ X[:, 0]
    # make row 1 dominant by 100, and rows 3, 5 far below
    pairs = [
        (1, 100.0 - float(resp[1])),
        (3, -10.0 - float(resp[3])),
        (5, -20.0 - float(resp[5])),
    ]
    _verify(client, 0, 0, pairs)
    page = _wait_for_verification(client)
    assert "k = 1" in page
    assert "k − 1" not in page
    assert "不给出 k−1 的漏词清单" in page


def test_merged_scores_come_from_zip_factors(client, problem):
    """On an unprotected activation the merged ranking can differ from W."""
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 4, [0, 1])  # 4 is above minimum; residual remains
    B_zip, A_zip = _zip_factors(client)

    for activation in range(2, X.shape[1]):
        x = X[:, activation]
        original = W @ x
        merged = B_zip @ (A_zip @ x)
        pairs = [(row, 0.0) for row in range(6)]
        top = int(np.argmax(original))
        expected_k = int(np.count_nonzero(merged >= merged[top]))
        _verify(client, activation, 0, pairs)
        page = _wait_for_verification(client)
        assert f"k = {expected_k}" in page
        if expected_k > 1:
            assert f"激活 #{activation}" in page
            expected_missed = [
                row
                for row in range(6)
                if original[row] >= original[top]
                and np.count_nonzero(merged >= merged[row]) > expected_k - 1
            ]
            assert "输出行 " + "、".join(str(r) for r in expected_missed) in page
        break


def test_invalid_candidate_input_reports_reason(client, problem):
    _upload(client, problem)
    _merge(client, 6, [0])
    _verify(client, 0, 0, [(0, 0.0), (99, 0.0)])
    page = client.get("/").text
    assert "核定未完成" in page
    assert "超出范围" in page
    assert "k =" not in page


def test_candidate_pair_with_one_side_missing_is_reported(client, problem):
    _upload(client, problem)
    _merge(client, 6, [0])
    client.post(
        "/verify",
        data={
            "activation_index": "0",
            "epsilon": "0",
            "candidate_row": ["0", "1", ""],
            "candidate_base": ["0", "", ""],
        },
        follow_redirects=False,
    )
    assert "缺少底座分数" in client.get("/").text


def test_reupload_withdraws_conclusion(client, problem):
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    assert "k =" in _wait_for_verification(client)

    _upload(client, problem, X=X + 0.5)
    page = client.get("/").text
    assert "k =" not in page
    assert "输入已变更，尚未重算" in page


def test_successful_repackage_withdraws_conclusion(client, problem):
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    page = _wait_for_verification(client)
    assert "k = 2" in page
    assert "第 1 版" in page

    _merge(client, 4, [2])  # success -> generation 2, new ZIP
    page = client.get("/").text
    assert "k = 2" not in page
    assert "第 2 版" in page


def test_failed_repackage_keeps_conclusion(client, problem):
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    assert "k = 2" in _wait_for_verification(client)

    _merge(client, 0, [0, 1, 2, 3, 4, 5])  # infeasible; package unchanged
    page = client.get("/").text
    assert "k = 2" in page
    assert "第 1 版" in page


def test_inflight_solve_is_discarded_after_upload(client, problem, monkeypatch):
    import threading

    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])

    started = threading.Event()
    release = threading.Event()
    real_evaluate = __import__("app.verification", fromlist=["evaluate"]).evaluate

    def slow_evaluate(*args, **kwargs):
        started.set()
        release.wait(timeout=10)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr("app.main.verification.evaluate", slow_evaluate)
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    assert started.wait(timeout=5)

    # Reload inputs while the old solve is blocked.  It must not be published.
    _upload(client, problem, X=X + 0.25)
    release.set()
    time.sleep(0.3)
    page = client.get("/").text
    assert "k =" not in page
    assert "正在用当前这版包求解" not in page


def test_corrupt_package_reports_failure_reason(client, problem):
    """If the stored ZIP cannot be read back, the solve fails loudly."""
    from app.store import SessionStore

    B1, A1, B2, A2, X = problem
    _upload(client, problem)
    _merge(client, 6, [0])
    # Corrupt the published bytes behind the session's back.
    cookie = client.cookies.get("lora_merge_session")
    store: SessionStore = app.state.store
    session = store.get(cookie)
    session.published.data = b"not a zip"
    _verify(client, 0, 0, [(0, 0.0), (1, 0.0)])
    page = _wait_for_verification(client)
    assert "核定未完成" in page
    assert "无法从下载包读回因子" in page
    assert "k =" not in page


def test_verification_is_bound_to_package_hash(client, problem):
    from app.store import SessionStore

    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    assert "k = 2" in _wait_for_verification(client)

    store: SessionStore = app.state.store
    session = store.get(client.cookies.get("lora_merge_session"))
    # Swap in different ZIP bytes while keeping the same generation number.
    with session.lock:
        session.published.data = session.published.data + b"\0"
        assert "current" in session.summary()["verification"]
        assert session.summary()["verification"]["current"] is False


def test_invalid_epsilon_is_reported(client, problem):
    _upload(client, problem)
    _merge(client, 6, [0])
    client.post(
        "/verify",
        data={
            "activation_index": "0",
            "epsilon": "-1",
            "candidate_row": ["0", "1"],
            "candidate_base": ["0", "0"],
        },
        follow_redirects=False,
    )
    assert "浮动上限 ε 不能为负数" in client.get("/").text


def test_duplicate_candidate_rows_rejected(client, problem):
    _upload(client, problem)
    _merge(client, 6, [0])
    _verify(client, 0, 0, [(2, 0.0), (2, 1.0)])
    assert "重复" in client.get("/").text


def test_new_verification_supersedes_inflight_one(client, problem, monkeypatch):
    import threading

    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])

    started = threading.Event()
    release = threading.Event()
    real_evaluate = __import__("app.verification", fromlist=["evaluate"]).evaluate
    calls = {"n": 0}

    def slow_evaluate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            release.wait(timeout=10)
            # First attempt gives an obviously different k from the second.
            args = list(args)
            args[4] = [(0, 0.0), (1, 0.0)]
            args[5] = 1e9
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr("app.main.verification.evaluate", slow_evaluate)
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    assert started.wait(timeout=5)

    # Second attempt while the first is blocked.
    _verify(client, 1, 1e9, [(0, 0.0), (1, 0.0)])
    release.set()
    time.sleep(0.3)
    page = client.get("/").text
    assert "ε = 1e+09" in page
    assert "激活 #1" in page


def test_invalid_retry_does_not_withdraw_previous_conclusion(client, problem):
    """A malformed retry fails on its own; the earlier valid result stays."""
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    _upload(client, problem)
    _merge(client, 6, [0])
    resp = W @ X[:, 0]
    _verify(client, 0, 0, [(0, -float(resp[0])), (1, -float(resp[1]))])
    assert "k = 2" in _wait_for_verification(client)

    _verify(client, 0, 0, [(0, 0.0), (99, 0.0)])  # row out of range
    page = client.get("/").text
    assert "核定未完成" in page
    assert "k = 2" in page


def test_input_error_clears_result_when_inputs_are_missing(client):
    client.post(
        "/verify",
        data={
            "activation_index": "0",
            "epsilon": "0",
            "candidate_row": ["0", "1"],
            "candidate_base": ["0", "0"],
        },
        follow_redirects=False,
    )
    assert "请先上传两份适配器与激活向量" in client.get("/").text


def test_verify_rejects_stale_package(client, problem):
    B1, A1, B2, A2, X = problem
    _upload(client, problem)
    _merge(client, 6, [0])
    _upload(client, problem, X=X + 0.25)  # package now stale
    _verify(client, 0, 0, [(0, 0.0), (1, 0.0)])
    page = client.get("/").text
    assert "输入已变更、尚未重算" in page
    assert "k =" not in page
