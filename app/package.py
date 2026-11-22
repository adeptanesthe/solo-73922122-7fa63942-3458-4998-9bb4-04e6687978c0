"""Package construction for a successful merge.

A package is a ZIP archive with exactly the artefacts a deployment engineer
needs:

* ``B.npy`` / ``A.npy`` -- the merged factors,
* ``manifest.json`` -- the rank limit and protected activation set *this*
  package was built with, plus metrics and hashes,
* ``README.txt`` -- short human-readable description.

:func:`build_package` re-opens its own archive and verifies the exported
factors reproduce the protected response before returning it.  A package that
fails that check is never published.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np

SCHEMA = "lora-merge-package/1"


class PackageVerificationError(RuntimeError):
    """The exported factors failed their own response self-test."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_vector(vector: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(vector, dtype=np.float64).tobytes())


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array, dtype=np.float64), allow_pickle=False)
    return buffer.getvalue()


def protected_manifest(X: np.ndarray, indices) -> list[dict]:
    return [
        {
            "index": int(i),
            "sha256": _sha256_vector(X[:, int(i)]),
            "norm": float(np.linalg.norm(X[:, int(i)])),
        }
        for i in indices
    ]


def build_package(result: dict, X: np.ndarray, generation: int) -> tuple[bytes, dict]:
    """Serialize the merge result and verify the exported factors.

    Returns ``(zip_bytes, manifest)``.  The caller publishes only if this
    function returns; any verification failure raises.
    """
    B = np.asarray(result["B"], dtype=np.float64)
    A = np.asarray(result["A"], dtype=np.float64)
    rank_limit = int(result["rank_limit"])
    protected = list(result["protected_indices"])

    if B.shape[1] != A.shape[0]:
        raise PackageVerificationError("因子共享维度不一致")
    if B.shape[1] > rank_limit:
        raise PackageVerificationError("因子共享维度超过秩上限")
    if not protected:
        raise PackageVerificationError("未记录保护项")
    if not 1 <= B.shape[0] <= 32 or not 1 <= A.shape[1] <= 32:
        raise PackageVerificationError("导出因子的层维度超出 1..32")

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation": int(generation),
        "rank_limit": rank_limit,
        "shared_dim": int(B.shape[1]),
        "protected_activations": protected_manifest(X, protected),
        "protected_activation_indices": protected,
        "layer": {
            "input_dim": int(A.shape[1]),
            "output_dim": int(B.shape[0]),
        },
        "factor_shapes": {"B": list(B.shape), "A": list(A.shape)},
        "metrics": {
            "response_max_abs_error_on_protected": float(
                result["response_max_abs_error"]
            ),
            "frobenius_distance": float(result["frobenius_distance"]),
            "frobenius_relative": float(result["frobenius_relative"]),
            "original_update_norm": float(result["original_update_norm"]),
            "minimum_feasible_rank": int(result["minimum_feasible_rank"]),
        },
        "sha256": {"B.npy": "", "A.npy": ""},
    }

    B_bytes = _npy_bytes(B)
    A_bytes = _npy_bytes(A)
    manifest["sha256"] = {"B.npy": _sha256_bytes(B_bytes), "A.npy": _sha256_bytes(A_bytes)}

    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    readme = (
        "LoRA merge package\n"
        "==================\n"
        f"rank_limit: {rank_limit}\n"
        f"protected activation indices (0-based): {protected}\n"
        "B.npy: shape {B}\n"
        "A.npy: shape {A}\n"
        "manifest.json: machine-readable metadata and hashes\n\n"
        "The factors satisfy B @ (A @ X_protected) == W @ X_protected within\n"
        "floating-point tolerance, where W is the sum of the original adapters.\n"
    ).format(B=list(B.shape), A=list(A.shape))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("B.npy", B_bytes)
        zf.writestr("A.npy", A_bytes)
        zf.writestr("manifest.json", manifest_bytes)
        zf.writestr("README.txt", readme.encode("utf-8"))
    package = buffer.getvalue()

    _verify_package(package, result["W"], X, protected, rank_limit)
    return package, json.loads(manifest_bytes)


def _verify_package(zip_bytes: bytes, W, X, protected, rank_limit) -> None:
    """Re-open the archive and self-test the exported factors."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            required = {"B.npy", "A.npy", "manifest.json"}
            if not required <= names:
                raise PackageVerificationError(f"打包缺少文件：{sorted(required - names)}")
            B = np.load(io.BytesIO(zf.read("B.npy")), allow_pickle=False)
            A = np.load(io.BytesIO(zf.read("A.npy")), allow_pickle=False)
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    except (zipfile.BadZipFile, ValueError, KeyError, OSError) as exc:
        raise PackageVerificationError(f"打包产物无法重新读取：{exc}") from exc

    if B.ndim != 2 or A.ndim != 2:
        raise PackageVerificationError("导出的因子不是二维矩阵")
    if B.shape[1] != A.shape[0] or B.shape[1] > rank_limit:
        raise PackageVerificationError("导出的因子共享维度不合法")
    if manifest.get("rank_limit") != rank_limit:
        raise PackageVerificationError("manifest 中的秩上限与本次不一致")
    if manifest.get("protected_activation_indices") != list(protected):
        raise PackageVerificationError("manifest 中的保护项与本次不一致")

    indices = np.asarray(protected, dtype=int)
    target = W @ X[:, indices]
    merged = B @ (A @ X[:, indices])
    scale = max(1.0, float(np.linalg.norm(W, "fro")) * float(np.linalg.norm(X[:, indices], "fro")))
    error = float(np.max(np.abs(merged - target))) if target.size else 0.0
    if not np.isfinite(error) or error > 1e-9 * scale:
        raise PackageVerificationError(
            f"导出的因子未通过响应自测：偏差 {error:.3e}"
        )
