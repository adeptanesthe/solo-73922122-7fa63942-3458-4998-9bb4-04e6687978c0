"""Candidate-count verification for the merged package.

The edge-side model uses the merged LoRA update as a cheap *filter*: it keeps
the top ``k`` candidate words, and the original model re-ranks those words
afterwards.  Filtering is only safe if every word that the *original* update
considers best survives the filter.

For one activation vector ``x`` a candidate word is defined by an output row
``row`` and a fixed base score ``base`` (for example a logit produced by the
base model).  Its score is::

    score(word) = base + LoRA_response[row]

where the response comes either from the combined original update
``W = B1 A1 + B2 A2`` or from the factors read back out of the published ZIP.

Floating tolerance
------------------
End-side arithmetic is not exact, so a single non-negative ``epsilon`` is
shared by every coordinate (every candidate score):

* a word is an *original top word* when its original score is within
  ``epsilon`` of the best original score;
* in the merged ranking two words whose scores differ by at most ``epsilon``
  are tied.  A tie is resolved **against** the word being ranked: its rank is
  the number of words whose merged score is greater than, or within
  ``epsilon`` of, its own score.  That is the most unfavourable position the
  word can occupy.

The minimal number of candidates is then::

    k = max over original top words of their most-unfavourable merged rank

and every original top word is guaranteed a place in the merged top ``k``.
Ranks are computed **among the submitted candidate words** (the only rows whose
base scores are known); each candidate is one word.  If ``k > 1`` the words
that would be lost with only ``k - 1`` candidates are reported explicitly.

Scoring is intentionally a pure function of its inputs.  The web layer binds
the inputs to one published package generation; this module only computes.
"""

from __future__ import annotations

import math

import numpy as np

MIN_CANDIDATES = 2
MAX_CANDIDATES = 8


class CandidateError(ValueError):
    """Invalid candidate-verification input, safe to show to the user."""


def parse_epsilon(raw) -> float:
    """Parse the shared floating upper bound ``epsilon``."""
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise CandidateError("请填写浮动上限 ε")
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"浮动上限 ε 必须是数字，实际为 {text!r}") from exc
    if not math.isfinite(value):
        raise CandidateError("浮动上限 ε 必须是有限数值")
    if value < 0:
        raise CandidateError("浮动上限 ε 不能为负数")
    return value


def parse_activation_index(raw, activation_count: int) -> int:
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise CandidateError("请选择一条已载入的激活")
    try:
        index = int(text)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"激活序号 {text!r} 不是整数") from exc
    if index < 0 or index >= activation_count:
        raise CandidateError(
            f"激活序号 {index} 超出范围（当前共 {activation_count} 条激活）"
        )
    return index


def parse_candidate_pairs(row_values, score_values) -> list[tuple[int, float]]:
    """Turn the paired form columns into ``[(output_row, base_score), ...]``.

    Fully blank rows from the fixed-size HTML form are ignored.  A row that has
    only one of the two fields filled is reported as an error rather than
    silently dropped.
    """
    rows = list(row_values or [])
    scores = list(score_values or [])
    if len(rows) != len(scores):
        raise CandidateError(
            f"候选行号与底座分数的数量不一致（{len(rows)} vs {len(scores)}）"
        )

    pairs: list[tuple[int, float]] = []
    for position, (row_raw, score_raw) in enumerate(zip(rows, scores), start=1):
        row_text = "" if row_raw is None else str(row_raw).strip()
        score_text = "" if score_raw is None else str(score_raw).strip()
        if not row_text and not score_text:
            continue
        if not row_text:
            raise CandidateError(f"第 {position} 个候选缺少输出行号")
        if not score_text:
            raise CandidateError(f"第 {position} 个候选缺少底座分数")
        try:
            row = int(row_text)
        except (TypeError, ValueError) as exc:
            raise CandidateError(
                f"第 {position} 个候选的输出行号 {row_text!r} 不是整数"
            ) from exc
        if row < 0:
            raise CandidateError(f"第 {position} 个候选的输出行号不能为负数")
        try:
            base = float(score_text)
        except (TypeError, ValueError) as exc:
            raise CandidateError(
                f"第 {position} 个候选的底座分数 {score_text!r} 不是数字"
            ) from exc
        if not math.isfinite(base):
            raise CandidateError(f"第 {position} 个候选的底座分数必须是有限数值")
        pairs.append((row, base))

    if len(pairs) < MIN_CANDIDATES:
        raise CandidateError(
            f"候选词至少需要 {MIN_CANDIDATES} 个，实际只填写了 {len(pairs)} 个"
        )
    if len(pairs) > MAX_CANDIDATES:
        raise CandidateError(
            f"候选词最多 {MAX_CANDIDATES} 个，实际填写了 {len(pairs)} 个"
        )
    seen: set[int] = set()
    for row, _ in pairs:
        if row in seen:
            raise CandidateError(
                f"候选输出行号 {row} 重复；同一个词只能填写一次"
            )
        seen.add(row)
    return pairs


def _as_matrix(name: str, value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"{name} 不是数值矩阵") from exc
    if arr.ndim != 2:
        raise CandidateError(f"{name} 必须是二维矩阵")
    if arr.size and not np.all(np.isfinite(arr)):
        raise CandidateError(f"{name} 含有 NaN 或无穷值")
    return arr


def evaluate(
    W,
    x,
    B_zip,
    A_zip,
    pairs: list[tuple[int, float]],
    epsilon: float,
) -> dict:
    """Score candidates and compute the minimal safe candidate count.

    ``W`` is the combined original update, ``x`` one activation column and
    ``B_zip``/``A_zip`` the factors read back from the published ZIP.
    """
    W = _as_matrix("原更新 W", W)
    B_zip = _as_matrix("ZIP 中的 B", B_zip)
    A_zip = _as_matrix("ZIP 中的 A", A_zip)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 2 and x_arr.shape[1] == 1:
        x_arr = x_arr[:, 0]
    if x_arr.ndim != 1:
        raise CandidateError("激活必须是单个列向量")
    if x_arr.size and not np.all(np.isfinite(x_arr)):
        raise CandidateError("激活含有 NaN 或无穷值")
    if not math.isfinite(float(epsilon)) or float(epsilon) < 0:
        raise CandidateError("浮动上限 ε 必须是非负有限数值")

    if B_zip.shape[1] != A_zip.shape[0]:
        raise CandidateError("ZIP 中 B、A 的共享维度不一致")
    if B_zip.shape[0] != W.shape[0] or A_zip.shape[1] != W.shape[1]:
        raise CandidateError(
            "ZIP 因子的层维度与两份原适配器的合计更新不一致："
            f"W 为 {W.shape[0]}×{W.shape[1]}，"
            f"B 为 {B_zip.shape[0]}×{B_zip.shape[1]}，"
            f"A 为 {A_zip.shape[0]}×{A_zip.shape[1]}"
        )
    if W.shape[1] != x_arr.shape[0]:
        raise CandidateError(
            f"激活维度 {x_arr.shape[0]} 与层输入维度 {W.shape[1]} 不一致"
        )

    out_dim = int(W.shape[0])
    if not pairs:
        raise CandidateError(f"候选词至少需要 {MIN_CANDIDATES} 个")
    for row, _ in pairs:
        if row < 0 or row >= out_dim:
            raise CandidateError(
                f"候选输出行号 {row} 超出范围（层输出维度为 {out_dim}）"
            )

    response_original = W @ x_arr
    # Score exactly as the package exports it: B (A x), not a recombined W.
    response_merged = B_zip @ (A_zip @ x_arr)
    if not np.all(np.isfinite(response_original)):
        raise CandidateError("原版评分出现非有限值，请检查适配器与激活")
    if not np.all(np.isfinite(response_merged)):
        raise CandidateError("合并版评分出现非有限值，请检查下载包中的因子")

    eps = float(epsilon)
    rows = [int(row) for row, _ in pairs]
    bases = [float(base) for _, base in pairs]
    original = np.asarray(
        [bases[i] + float(response_original[rows[i]]) for i in range(len(rows))],
        dtype=np.float64,
    )
    merged = np.asarray(
        [bases[i] + float(response_merged[rows[i]]) for i in range(len(rows))],
        dtype=np.float64,
    )
    if not (np.all(np.isfinite(original)) and np.all(np.isfinite(merged))):
        raise CandidateError("候选词评分出现非有限值，请检查输入")

    max_original = float(np.max(original))
    top_flags = [bool(original[i] >= max_original - eps) for i in range(len(rows))]

    # Most unfavourable merged rank: words strictly ahead count, and so do
    # words tied within epsilon (including the word itself).
    ranks = [
        int(np.count_nonzero(merged >= merged[i] - eps)) for i in range(len(rows))
    ]

    top_positions = [i for i in range(len(rows)) if top_flags[i]]
    k = max(ranks[i] for i in top_positions)

    previous_k = k - 1
    if k > 1:
        missed_positions = [i for i in top_positions if ranks[i] > previous_k]
    else:
        missed_positions = []

    candidates = [
        {
            "position": i + 1,
            "row": rows[i],
            "base_score": bases[i],
            "original_score": float(original[i]),
            "merged_score": float(merged[i]),
            "rank": ranks[i],
            "is_original_top": top_flags[i],
        }
        for i in range(len(rows))
    ]
    return {
        "k": int(k),
        "previous_k": int(previous_k),
        "epsilon": eps,
        "candidate_count": len(rows),
        "candidates": candidates,
        "top_rows": [rows[i] for i in top_positions],
        "max_original_score": max_original,
        "missed_rows": [rows[i] for i in missed_positions],
        "missed_candidates": [candidates[i] for i in missed_positions],
    }
