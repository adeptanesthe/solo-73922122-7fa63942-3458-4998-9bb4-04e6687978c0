from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import factorization
from .package import PackageVerificationError, build_package
from .parsing import UploadError, parse_activations, parse_uploaded_adapter
from .store import PublishedPackage, Session, SessionStore

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
    response = templates.TemplateResponse(
        request,
        "index.html",
        {"session": session.summary(), "activations": _activation_rows(session), **extra},
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
