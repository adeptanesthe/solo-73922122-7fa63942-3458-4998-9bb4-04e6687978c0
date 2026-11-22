import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import factorization as F  # noqa: E402
from app.package import PackageVerificationError, build_package  # noqa: E402
from app.parsing import (  # noqa: E402
    UploadError,
    parse_activations,
    parse_adapter_bundle,
    parse_matrix_file,
    parse_uploaded_adapter,
)


def _problem(seed=1):
    rng = np.random.default_rng(seed)
    B1 = rng.normal(size=(6, 3))
    A1 = rng.normal(size=(3, 5))
    B2 = rng.normal(size=(6, 2))
    A2 = rng.normal(size=(2, 5))
    X = rng.normal(size=(5, 4))
    return B1, A1, B2, A2, X


def _npy(arr):
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def test_parse_npy_bundle_and_factors():
    B1, A1, B2, A2, X = _problem()
    B, A = parse_adapter_bundle(_npz(lora_B=B1, lora_A=A1), "a1.npz")
    assert np.array_equal(B, B1) and np.array_equal(A, A1)
    B2p, A2p = parse_uploaded_adapter(
        (None, None), (_npy(B2), "b.npy"), (_npy(A2), "a.npy")
    )
    assert np.array_equal(B2p, B2) and np.array_equal(A2p, A2)


def test_parse_peft_style_keys():
    B1, A1, _, _, _ = _problem()
    payload = _npz(**{
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": B1,
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": A1,
    })
    B, A = parse_adapter_bundle(payload, "peft.npz")
    assert np.array_equal(B, B1) and np.array_equal(A, A1)


def test_parse_json_bundle_and_csv():
    B1, A1, _, _, X = _problem()
    payload = json.dumps({"B": B1.tolist(), "A": A1.tolist()}).encode()
    B, A = parse_adapter_bundle(payload, "a1.json")
    assert np.allclose(B, B1) and np.allclose(A, A1)

    csv = "\n".join(",".join(f"{v:.17g}" for v in row) for row in X.T).encode()
    parsed = parse_activations(csv, "x.csv", 5)
    assert np.allclose(parsed, X)


def test_activation_transpose_auto_orientation():
    _, _, _, _, X = _problem()
    payload = _npy(X.T)
    parsed = parse_activations(payload, "x.npy", 5)
    assert np.allclose(parsed, X)


def test_ambiguous_upload_rejected():
    B1, A1, _, _, _ = _problem()
    with pytest.raises(UploadError):
        parse_uploaded_adapter(
            (_npz(B=B1, A=A1), "bundle.npz"),
            (_npy(B1), "b.npy"),
            (_npy(A1), "a.npy"),
        )
    with pytest.raises(UploadError):
        parse_uploaded_adapter((None, None), (_npy(B1), "b.npy"), (None, None))


def test_malformed_payloads_rejected():
    with pytest.raises(UploadError):
        parse_matrix_file(b"not a matrix", "x.npy")
    with pytest.raises(UploadError):
        parse_adapter_bundle(_npy(np.zeros((2, 2))), "a.npy")


def test_package_self_check_and_manifest():
    B1, A1, B2, A2, X = _problem(3)
    result = F.merge(B1, A1, B2, A2, X, 4, [0, 2])
    package, manifest = build_package(result, X, 7)
    with zipfile.ZipFile(io.BytesIO(package)) as zf:
        assert set(zf.namelist()) == {"A.npy", "B.npy", "manifest.json", "README.txt"}
        shipped = json.loads(zf.read("manifest.json"))
        B = np.load(io.BytesIO(zf.read("B.npy")))
        A = np.load(io.BytesIO(zf.read("A.npy")))
    assert shipped["rank_limit"] == 4
    assert shipped["protected_activation_indices"] == [0, 2]
    assert shipped["shared_dim"] <= 4
    assert np.abs(B @ (A @ X[:, [0, 2]]) - result["W"] @ X[:, [0, 2]]).max() < 1e-10
    assert manifest == shipped


def test_package_refuses_wrong_factors():
    B1, A1, B2, A2, X = _problem(4)
    result = F.merge(B1, A1, B2, A2, X, 4, [0, 1])
    tampered = dict(result)
    tampered["B"] = result["B"] * 0.5
    with pytest.raises(PackageVerificationError):
        build_package(tampered, X, 1)
