"""Parsing of uploaded LoRA factors and activation matrices.

Supported container formats:

* ``.npy`` -- one matrix.
* ``.npz`` -- named matrices; for an adapter, keys resolving to ``A`` and
  ``B`` (e.g. ``A``, ``B``, ``lora_A``, ``lora_B``, ``...lora_A.weight``).
* ``.json`` -- nested lists, or an object containing the matrices.
* ``.csv`` / ``.txt`` -- numeric text, comma or whitespace separated.

For an adapter the two factors may be uploaded either as one archive/JSON
object or as two separate files.  The conventions are ``B`` of shape
``(out_dim, r)`` and ``A`` of shape ``(r, in_dim)``.
"""

from __future__ import annotations

import io
import json
import re
import zipfile

import numpy as np

MAX_FILE_BYTES = 8 * 1024 * 1024


class UploadError(ValueError):
    """An uploaded file could not be interpreted."""


def _check_size(data: bytes, filename: str) -> None:
    if not data:
        raise UploadError(f"{filename}: 文件为空")
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"{filename}: 文件超过 {MAX_FILE_BYTES // (1024 * 1024)} MB 限制")


def _to_float_array(value, filename: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UploadError(f"{filename}: 内容不是数值数组") from exc
    if arr.ndim == 0:
        raise UploadError(f"{filename}: 内容是标量，需要二维矩阵")
    return arr


def _load_npy(data: bytes, filename: str) -> np.ndarray:
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:  # ValueError, EOFError, ...
        raise UploadError(f"{filename}: 无法解析 .npy 文件（{exc}）") from exc
    return _to_float_array(arr, filename)


def _load_npz(data: bytes, filename: str) -> dict:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            return {key: _to_float_array(npz[key], f"{filename}:{key}") for key in npz.files}
    except UploadError:
        raise
    except Exception as exc:
        raise UploadError(f"{filename}: 无法解析 .npz 文件（{exc}）") from exc


def _load_json(data: bytes, filename: str) -> object:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UploadError(f"{filename}: 无法解析 JSON（{exc}）") from exc


def _load_text(data: bytes, filename: str) -> np.ndarray:
    text = data.decode("utf-8", errors="strict") if _is_utf8(data) else None
    if text is None:
        raise UploadError(f"{filename}: 文本文件必须是 UTF-8 编码")
    for delimiter in (",", None):
        try:
            arr = np.loadtxt(io.StringIO(text), delimiter=delimiter)
        except ValueError:
            continue
        if arr.size:
            return _to_float_array(arr, filename)
    raise UploadError(f"{filename}: 无法解析为数值矩阵（支持逗号或空白分隔）")


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _sniff(data: bytes, filename: str) -> str:
    """Return 'npy', 'npz', 'json' or 'text'."""
    name = (filename or "").lower()
    if data[:6] == b"\x93NUMPY":
        return "npy"
    if data[:2] == b"PK" and (name.endswith(".npz") or _zip_has_npy(data)):
        return "npz"
    if name.endswith(".npy"):
        return "npy"
    if name.endswith(".npz"):
        return "npz"
    if name.endswith(".json"):
        return "json"
    stripped = data.lstrip()
    if stripped[:1] in (b"{", b"["):
        return "json"
    return "text"


def _zip_has_npy(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return any(item.filename.endswith(".npy") for item in zf.infolist())
    except zipfile.BadZipFile:
        return False


def load_any(data: bytes, filename: str):
    """Load a file into either an ndarray or a dict of arrays."""
    _check_size(data, filename)
    kind = _sniff(data, filename)
    if kind == "npy":
        return _load_npy(data, filename)
    if kind == "npz":
        return _load_npz(data, filename)
    if kind == "json":
        payload = _load_json(data, filename)
        if isinstance(payload, dict):
            return {
                str(key): _to_float_array(value, f"{filename}:{key}")
                for key, value in payload.items()
            }
        return _to_float_array(payload, filename)
    return _load_text(data, filename)


def _normalise_key(key: str) -> str:
    """Reduce a PEFT-style key to the bare factor name where possible."""
    key = key.strip().lower()
    key = re.sub(r"\.(weight|default)$", "", key)
    parts = re.split(r"[./]", key)
    return parts[-1] if parts else key


def _pick_factor(payload: dict, which: str, filename: str) -> np.ndarray:
    """Find the ``A`` or ``B`` array in a mapping of arrays."""
    targets = {which, f"lora_{which}", f"{which}_lora", f"{which}_weight"}
    matches = [key for key in payload if _normalise_key(key) in targets]
    if len(matches) == 1:
        return payload[matches[0]]
    if len(matches) > 1:
        raise UploadError(
            f"{filename}: 找到多个疑似因子 {which} 的键 {sorted(matches)}；"
            "请上传只含单层权重的文件，或分别上传 B 与 A 文件"
        )
    raise UploadError(
        f"{filename}: 未找到因子 {which}（可用键示例：{which}、lora_{which}、"
        f"...lora_{which}.weight）；现有键 {sorted(payload)}"
    )


def parse_adapter_bundle(data: bytes, filename: str):
    """Parse a single file that contains both ``B`` and ``A``."""
    payload = load_any(data, filename)
    if not isinstance(payload, dict):
        raise UploadError(
            f"{filename}: 需要同时包含 A、B 两个因子的文件（.npz/.json），"
            "或分别上传 B、A 两个 .npy 文件"
        )
    return _pick_factor(payload, "b", filename), _pick_factor(payload, "a", filename)


def parse_matrix_file(data: bytes, filename: str) -> np.ndarray:
    payload = load_any(data, filename)
    if isinstance(payload, dict):
        if len(payload) != 1:
            raise UploadError(
                f"{filename}: 期望单个矩阵，但文件包含多个数组 {sorted(payload)}"
            )
        payload = next(iter(payload.values()))
    if payload.ndim != 2:
        raise UploadError(f"{filename}: 期望二维矩阵，实际为 {payload.ndim} 维")
    return payload


def parse_activations(data: bytes, filename: str, in_dim: int) -> np.ndarray:
    payload = load_any(data, filename)
    if isinstance(payload, dict):
        for preferred in ("X", "x", "activations", "inputs", "data"):
            for key in payload:
                if _normalise_key(key) == preferred:
                    payload = payload[key]
                    break
            else:
                continue
            break
        else:
            if len(payload) == 1:
                payload = next(iter(payload.values()))
            else:
                raise UploadError(
                    f"{filename}: 无法确定激活矩阵，请使用键 X/activations，"
                    f"现有键 {sorted(payload)}"
                )
    if payload.ndim == 1:
        payload = payload.reshape(-1, 1)
    if payload.ndim != 2:
        raise UploadError(f"{filename}: 激活矩阵必须是二维的")
    # Auto-orientation: rows are features by convention.  If that does not
    # match while the transpose does, accept the transposed layout.
    if payload.shape[0] != in_dim and payload.shape[1] == in_dim:
        payload = payload.T
    return payload


def parse_uploaded_adapter(
    bundle: tuple[bytes | None, str | None],
    b_factor: tuple[bytes | None, str | None],
    a_factor: tuple[bytes | None, str | None],
):
    """Resolve an adapter from an optional bundle plus optional factor files.

    Each argument is a ``(data, filename)`` pair as produced from the upload
    form.  Either a single bundle containing both factors or both individual
    factor files must be supplied.
    """
    bundle_data, bundle_name = bundle
    b_data, b_name = b_factor
    a_data, a_name = a_factor

    has_bundle = bundle_data is not None and bundle_name is not None
    has_b = b_data is not None and b_name is not None
    has_a = a_data is not None and a_name is not None

    if has_bundle and (has_b or has_a):
        raise UploadError(
            "适配器只能选一种上传方式：一个整包文件（含 A、B），或分别上传 B 与 A"
        )
    if has_bundle:
        return parse_adapter_bundle(bundle_data, bundle_name)
    if not (has_b and has_a):
        raise UploadError(
            "适配器文件不完整：请上传含 A、B 的整包，或同时提供 B 与 A 两个文件"
        )
    return parse_matrix_file(b_data, b_name), parse_matrix_file(a_data, a_name)
