from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import factorization, verification
from .package import (
    PackageVerificationError,
    build_package,
    load_factors_from_zip,
)
from .parsing import UploadError, parse_activations, parse_uploaded_adapter
from .store import (
    PublishedPackage,
    Session,
    SessionStore,
    VerificationResult,
)

SESSION_COOKIE = "lora_merge_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 12
MAX_RANK_LIMIT = 128


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = SessionStore()
    yield


app = FastAPI(title="LoRA 单层适配器合并", lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")


def _session(request: Request) -> Session:
    store: SessionStore = request.app.state.store
    session = store.get(request.cookies.get(SESSION_COOKIE, ""))
    if session is None:
        session = store.create()
    return session


def _attach_cookie(response, session: Session):
    response.set_cookie(
        SESSION_COOKIE,
        session.id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


def _activation_rows(session: Session):
    if session.X is None:
        return []
    rows = []
    for idx in range(session.X.shape[1]):
        column = session.X[:, idx]
        preview = ", ".join(f"{v:.3g}" for v in column[:4])
        if column.size > 4:
            preview += ", …"
        rows.append(
            {
                "index": idx,
                "preview": preview,
                "norm": float((column**2).sum() ** 0.5),
                "min": float(column.min()),
                "max": float(column.max()),
            }
        )
    return rows


def _render(request: Request, session: Session, status_code: int = 200, **extra):
    # Serialise against a worker thread publishing its outcome, so the page
    # never shows a half-updated verification state.
    with session.lock:
        summary = session.summary()
        activations = _activation_rows(session)
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "session": summary,
            "activations": activations,
            "verify_slots": verification.MAX_CANDIDATES,
            **extra,
        },
        status_code=status_code,
    )
    return _attach_cookie(response, session)


def _redirect(session: Session) -> RedirectResponse:
    return _attach_cookie(RedirectResponse("/", status_code=303), session)


async def _read(upload: UploadFile | None):
    if upload is None or not upload.filename:
        return None, None
    return await upload.read(), upload.filename


def _parse_rank_limit(raw: str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise factorization.DataError("秩上限必须是整数") from exc
    if value < 0 or value > MAX_RANK_LIMIT:
        raise factorization.DataError(f"秩上限必须位于 0..{MAX_RANK_LIMIT} 之间")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_verification(
    *,
    session: Session,
    job: int,
    epoch: int,
    generation: int,
    package_sha256: str,
    fingerprint: str,
    activation_index: int,
    epsilon: float,
    pairs,
    W,
    x,
    package_data: bytes,
) -> None:
    """Solve in a worker thread, then publish only if still current.

    The currency check is under the session lock and covers both the job id
    (a newer attempt supersedes this one) and the state epoch (inputs were
    reloaded or a new package was published).  A solve that lost the race is
    discarded, never displayed.
    """
    outcome = None
    error = None
    try:
        # Score against the factors actually written in the ZIP.
        B_zip, A_zip, manifest = load_factors_from_zip(package_data)
        if manifest.get("generation") != generation:
            raise PackageVerificationError(
                "下载包 manifest 中的包版本与本次核定的版本不一致"
            )
        outcome = verification.evaluate(W, x, B_zip, A_zip, pairs, epsilon)
    except (verification.CandidateError, PackageVerificationError) as exc:
        error = {
            "reason": "evaluation",
            "message": str(exc),
            "generation": generation,
            "created_at": _now(),
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash the worker
        error = {
            "reason": "internal",
            "message": f"核定求解出现内部错误：{exc}",
            "generation": generation,
            "created_at": _now(),
        }

    with session.lock:
        if session.verification_job != job:
            return  # superseded by a newer attempt
        if session.state_epoch != epoch:
            return  # inputs reloaded or package replaced mid-solve
        published = session.published
        if published is None or published.sha256 != package_sha256:
            # The bytes we scored are no longer downloadable; do not leave a
            # stale "solving" banner behind.
            session.verification_running = None
            session.verification = None
            session.verification_error = {
                "reason": "package",
                "message": "求解期间下载包已变化，本次结论已作废，请重新核定。",
                "created_at": _now(),
            }
            return
        session.verification_running = None
        if error is not None:
            session.verification = None
            session.verification_error = error
            return
        session.verification_error = None
        session.verification = VerificationResult(
            generation=generation,
            package_sha256=package_sha256,
            input_fingerprint=fingerprint,
            activation_index=int(activation_index),
            epsilon=float(epsilon),
            pairs=[(int(r), float(s)) for r, s in pairs],
            outcome=outcome,
            created_at=_now(),
        )


def _start_verification(
    session: Session,
    job: int,
    activation_index: int,
    epsilon: float,
    pairs: list[tuple[int, float]],
) -> None:
    """Snapshot inputs, bind them to the current package, and start solving.

    ``job`` must already be the current attempt id (from
    :meth:`Session.begin_verification_attempt`).  Must be called with the
    session lock held.
    """
    published = session.published
    assert published is not None
    epoch = session.state_epoch
    session.verification_draft = {
        "activation_index": int(activation_index),
        "epsilon": float(epsilon),
        "pairs": [[int(r), float(s)] for r, s in pairs],
        "generation": published.generation,
    }
    session.verification_running = job
    W = session.combined_update().copy()
    x = session.X[:, activation_index].copy()
    args = dict(
        session=session,
        job=job,
        epoch=epoch,
        generation=published.generation,
        package_sha256=published.sha256,
        fingerprint=session.input_fingerprint(),
        activation_index=activation_index,
        epsilon=epsilon,
        pairs=list(pairs),
        W=W,
        x=x,
        package_data=published.data,
    )
    thread = threading.Thread(
        target=_run_verification, kwargs=args, name="verify-candidates", daemon=True
    )
    thread.start()



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render(request, _session(request))


@app.post("/upload")
async def upload(
    request: Request,
    adapter1_bundle: UploadFile | None = File(None),
    adapter1_b: UploadFile | None = File(None),
    adapter1_a: UploadFile | None = File(None),
    adapter2_bundle: UploadFile | None = File(None),
    adapter2_b: UploadFile | None = File(None),
    adapter2_a: UploadFile | None = File(None),
    activations: UploadFile | None = File(None),
):
    session = _session(request)
    with session.lock:
        try:
            B1, A1 = parse_uploaded_adapter(
                await _read(adapter1_bundle),
                await _read(adapter1_b),
                await _read(adapter1_a),
            )
            B2, A2 = parse_uploaded_adapter(
                await _read(adapter2_bundle),
                await _read(adapter2_b),
                await _read(adapter2_a),
            )
            B1, A1, B2, A2 = factorization.validate_adapters(B1, A1, B2, A2)
            X_data, X_name = await _read(activations)
            if X_data is None:
                raise UploadError("请上传激活向量文件")
            X = parse_activations(X_data, X_name, int(A1.shape[1]))
            X = factorization.validate_activations(X, int(A1.shape[1]))
        except (UploadError, factorization.MergeError) as exc:
            session.last_error = {"kind": "upload", "message": str(exc)}
            session.last_notice = None
            return _redirect(session)

        session.B1, session.A1, session.B2, session.A2, session.X = B1, A1, B2, A2, X
        # New inputs: any earlier candidate-count conclusion no longer
        # describes "this batch of inputs"; running solves are discarded too.
        session.invalidate_verification("upload")
        in_dim, out_dim = int(X.shape[0]), int(B1.shape[0])
        session.protected_indices = [
            i for i in session.protected_indices if i < int(X.shape[1])
        ]
        session.last_error = None
        session.last_notice = {
            "kind": "upload",
            "message": f"已载入两份适配器（{in_dim}→{out_dim}）与 {X.shape[1]} 个激活向量",
        }
    return _redirect(session)


def _parse_protected(raw: list[str]) -> list[int]:
    indices = []
    for item in raw:
        try:
            indices.append(int(str(item).strip()))
        except (TypeError, ValueError) as exc:
            raise factorization.DataError(f"保护项下标 {item!r} 不是整数") from exc
    return indices


@app.post("/merge")
async def merge(
    request: Request,
    rank_limit: str = Form(""),
    protected: list[str] = Form(default=[]),
):
    session = _session(request)
    with session.lock:
        if not session.inputs_ready():
            session.last_error = {
                "kind": "state",
                "message": "请先上传两份适配器与激活向量，再计算合并。",
            }
            session.last_notice = None
            return _redirect(session)

        try:
            rank = _parse_rank_limit(rank_limit)
            selected = _parse_protected(protected)
            result = factorization.merge(
                session.B1,
                session.A1,
                session.B2,
                session.A2,
                session.X,
                rank,
                selected,
            )
            next_generation = (
                session.published.generation + 1 if session.published else 1
            )
            package, manifest = build_package(result, session.X, next_generation)
        except factorization.InfeasibleRankError as exc:
            session.rank_limit = exc.requested_rank
            session.protected_indices = sorted(set(selected))
            session.last_error = {
                "kind": "infeasible",
                "message": str(exc),
                "min_rank": exc.min_rank,
                "requested_rank": exc.requested_rank,
            }
            session.last_notice = None
            return _redirect(session)
        except (factorization.MergeError, PackageVerificationError) as exc:
            session.last_error = {"kind": "merge", "message": str(exc)}
            session.last_notice = None
            return _redirect(session)

        # Publish only now: the package has been built and self-verified.
        session.rank_limit = rank
        session.protected_indices = list(result["protected_indices"])
        session.published = PublishedPackage(
            data=package,
            manifest=manifest,
            generation=next_generation,
            rank_limit=manifest["rank_limit"],
            protected_indices=list(manifest["protected_activation_indices"]),
            input_fingerprint=session.input_fingerprint(),
        )
        # A successful re-package replaces the ZIP: old conclusions (stored or
        # in flight) described the previous package and are withdrawn.
        session.invalidate_verification("package")
        session.last_error = None
        session.last_notice = {
            "kind": "merge",
            "message": (
                f"已生成第 {next_generation} 版下载包：秩上限 {manifest['rank_limit']}，"
                f"保护 {len(manifest['protected_activation_indices'])} 个激活，"
                "导出因子已通过响应自测"
            ),
        }
    return _redirect(session)


@app.post("/verify")
async def verify(
    request: Request,
    activation_index: str = Form(""),
    epsilon: str = Form(""),
    candidate_row: list[str] = Form(default=[]),
    candidate_base: list[str] = Form(default=[]),
):
    session = _session(request)
    with session.lock:
        if not session.inputs_ready():
            session.verification_error = {
                "reason": "state",
                "message": "请先上传两份适配器与激活向量，再核定候选数。",
            }
            session.verification = None
            return _redirect(session)
        if session.published is None:
            session.verification_error = {
                "reason": "state",
                "message": "还没有已发布的下载包；请先成功计算一次合并，再核定候选数。",
            }
            session.verification = None
            return _redirect(session)
        if session.package_is_stale():
            session.verification_error = {
                "reason": "state",
                "message": (
                    "当前下载包是在这批输入之前生成的（输入已变更、尚未重算），"
                    "无法把结论标到它上面。请先重新计算合并，再核定候选数。"
                ),
            }
            session.verification = None
            return _redirect(session)

        try:
            index = verification.parse_activation_index(
                activation_index, int(session.X.shape[1])
            )
            eps = verification.parse_epsilon(epsilon)
            pairs = verification.parse_candidate_pairs(candidate_row, candidate_base)
            out_dim = int(session.B1.shape[0])
            for row, _ in pairs:
                if row >= out_dim:
                    raise verification.CandidateError(
                        f"候选输出行号 {row} 超出范围（层输出维度为 {out_dim}）"
                    )
        except verification.CandidateError as exc:
            # The request itself is malformed.  Keep any earlier valid result
            # (it still describes a real, unchanged package) and only report
            # why this retry did not produce a new one.
            session.verification_error = {
                "reason": "input",
                "message": str(exc),
                "created_at": _now(),
            }
            return _redirect(session)

        # Validation passed: only now supersede the previous conclusion.
        job = session.begin_verification_attempt()
        _start_verification(session, job, index, eps, pairs)
    return _redirect(session)


@app.get("/download")
async def download(request: Request):
    session = _session(request)
    with session.lock:
        published = session.published
        if published is None:
            return _render(
                request,
                session,
                status_code=404,
                download_error="还没有可下载的合并包，请先成功计算一次合并。",
            )
        data = published.data
        filename = f"lora-merge-gen{published.generation}.zip"
        generation = published.generation
        rank_limit = published.rank_limit
        protected_indices = list(published.protected_indices)
    return _attach_cookie(
        Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Package-Generation": str(generation),
                "X-Rank-Limit": str(rank_limit),
                "X-Protected-Indices": ",".join(str(i) for i in protected_indices),
            },
        ),
        session,
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
