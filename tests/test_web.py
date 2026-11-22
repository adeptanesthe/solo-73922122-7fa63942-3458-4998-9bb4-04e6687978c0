import io
import json
import sys
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
    rng = np.random.default_rng(42)
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


def _upload(client, problem):
    B1, A1, B2, A2, X = problem
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
    data = {"rank_limit": str(rank), "protected": [str(i) for i in protected]}
    return client.post("/merge", data=data, follow_redirects=False)


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LoRA 单层适配器合并" in response.text


def test_full_successful_flow(client, problem):
    B1, A1, B2, A2, X = problem
    W = B1 @ A1 + B2 @ A2
    assert _upload(client, problem).status_code == 303
    assert _merge(client, 5, [0, 3, 6]).status_code == 303

    page = client.get("/").text
    assert "已生成第 1 版下载包" in page

    response = client.get("/download")
    assert response.status_code == 200
    assert response.headers["x-rank-limit"] == "5"
    assert response.headers["x-protected-indices"] == "0,3,6"
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        B = np.load(io.BytesIO(zf.read("B.npy")))
        A = np.load(io.BytesIO(zf.read("A.npy")))
    assert manifest["rank_limit"] == 5
    assert manifest["protected_activation_indices"] == [0, 3, 6]
    assert B.shape[1] == A.shape[0] <= 5
    assert np.abs(B @ (A @ X[:, [0, 3, 6]]) - W @ X[:, [0, 3, 6]]).max() < 1e-10


def test_infeasible_merge_keeps_previous_package(client, problem):
    B1, A1, B2, A2, X = problem
    _upload(client, problem)
    _merge(client, 5, [0, 3, 6])
    first = client.get("/download")
    assert first.headers["x-package-generation"] == "1"

    response = _merge(client, 1, [0, 1, 2, 3, 4, 5])
    assert response.status_code == 303
    page = client.get("/").text
    assert "最低可行秩" in page
    assert "未生成新下载包" in page

    after = client.get("/download")
    assert after.status_code == 200
    assert after.content == first.content
    assert after.headers["x-rank-limit"] == "5"
    assert after.headers["x-protected-indices"] == "0,3,6"
    assert after.headers["x-package-generation"] == "1"


def test_merge_replaces_package_on_success(client, problem):
    _upload(client, problem)
    _merge(client, 6, [0, 1])
    first = client.get("/download")
    _merge(client, 3, [2, 4])
    second = client.get("/download")
    assert first.content != second.content
    assert second.headers["x-package-generation"] == "2"
    assert second.headers["x-rank-limit"] == "3"
    assert second.headers["x-protected-indices"] == "2,4"


def test_failed_merge_does_not_advance_generation(client, problem):
    _upload(client, problem)
    _merge(client, 1, [0, 1, 2, 3, 4, 5])  # infeasible
    page = client.get("/").text
    assert "还没有可下载的合并包" not in page
    response = client.get("/download")
    assert response.status_code == 404
    _merge(client, 3, [0])
    ok = client.get("/download")
    assert ok.headers["x-package-generation"] == "1"


def test_download_before_any_merge_is_404(client):
    assert client.get("/download").status_code == 404


def test_merge_without_upload_is_refused(client):
    _merge(client, 4, [0])
    page = client.get("/").text
    assert "请先上传两份适配器与激活向量" in page


def test_bad_rank_limit_rejected(client, problem):
    _upload(client, problem)
    _merge(client, -1, [0])
    assert "秩上限必须位于" in client.get("/").text
    _merge(client, 999, [0])
    assert "秩上限必须位于" in client.get("/").text


def test_non_integer_protected_index_is_page_error_not_422(client, problem):
    _upload(client, problem)
    response = client.post(
        "/merge", data={"rank_limit": "4", "protected": ["abc"]}, follow_redirects=False
    )
    assert response.status_code == 303
    assert "不是整数" in client.get("/").text
    assert client.get("/download").status_code == 404


def test_missing_protected_selection_rejected(client, problem):
    _upload(client, problem)
    client.post("/merge", data={"rank_limit": "4"}, follow_redirects=False)
    assert "至少需要勾选一个" in client.get("/").text


def test_upload_with_bad_dimension_reports_error(client, problem):
    B1, A1, B2, A2, X = problem
    response = client.post(
        "/upload",
        files={
            "adapter1_bundle": ("a1.npz", _npz(lora_B=B1, lora_A=A1), "application/zip"),
            "adapter2_b": ("b2.npy", _npy(B2), "application/octet-stream"),
            "adapter2_a": ("a2.npy", _npy(A2), "application/octet-stream"),
            "activations": ("x.npy", _npy(X[:3]), "application/octet-stream"),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "行数必须等于层输入维度" in client.get("/").text


def test_new_upload_marks_package_stale_until_recomputed(client, problem):
    _upload(client, problem)
    _merge(client, 4, [0, 1])
    first = client.get("/download")
    assert first.status_code == 200

    # Re-upload different activations; the package exists but is out of date.
    rng = np.random.default_rng(99)
    X2 = rng.normal(size=(5, 8))
    client.post(
        "/upload",
        files={
            "adapter1_bundle": ("a1.npz", _npz(lora_B=problem[0], lora_A=problem[1]), "application/zip"),
            "adapter2_b": ("b2.npy", _npy(problem[2]), "application/octet-stream"),
            "adapter2_a": ("a2.npy", _npy(problem[3]), "application/octet-stream"),
            "activations": ("x2.npy", _npy(X2), "application/octet-stream"),
        },
        follow_redirects=False,
    )
    page = client.get("/").text
    assert "输入已变更，尚未重算" in page
    # The old package is still served, still labelled with its own metadata.
    still = client.get("/download")
    assert still.content == first.content
    assert still.headers["x-rank-limit"] == "4"
    assert still.headers["x-protected-indices"] == "0,1"
