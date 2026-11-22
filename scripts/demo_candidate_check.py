#!/usr/bin/env python3
"""End-to-end demo of the candidate-count verification over HTTP.

Run the service first (Docker Compose, or locally):

    docker compose up --build          # http://localhost:8000
    python scripts/demo_candidate_check.py --base http://localhost:8000

The script uploads a deterministic toy layer, merges it, then exercises the
three key behaviours:

1. a tie on the merged side (worst-case rank makes k > 1 and reports the lost
   words for k - 1);
2. a clear winner (k = 1, no lost-word list);
3. re-uploading inputs (old conclusion withdrawn, stale package refuses a new
   verification).

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import io
import json
import re
import sys
import time
import urllib.request
import uuid
import zipfile

import numpy as np


def _npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _npz(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _multipart(fields, files):
    boundary = f"----demo{uuid.uuid4().hex}"
    chunks = []
    for name, value in fields:
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )
    for name, filename, data in files:
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )

    def request(self, path, data=None, headers=None, follow=True):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        opener = self.opener if follow else urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar), _NoRedirect()
        )
        try:
            with opener.open(req, timeout=30) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers

    def post_form(self, path, fields):
        body, ctype = _multipart(fields, [])
        return self.request(path, body, {"Content-Type": ctype})

    def post_upload(self, files):
        body, ctype = _multipart([], files)
        return self.request(path="/upload", data=body, headers={"Content-Type": ctype})

    def page(self):
        status, body, _ = self.request("/")
        return status, body.decode("utf-8")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _wait(client: Client, timeout=15.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, page = client.page()
        if "正在用当前这版包求解" not in page:
            return page
        time.sleep(0.2)
    raise RuntimeError("verification did not finish")


def _fixture():
    rng = np.random.default_rng(2024)
    B1 = rng.normal(size=(6, 3))
    A1 = rng.normal(size=(3, 5))
    B2 = rng.normal(size=(6, 2))
    A2 = rng.normal(size=(2, 5))
    X = rng.normal(size=(5, 8))
    return B1, A1, B2, A2, X


def _upload(client, B1, A1, B2, A2, X):
    return client.post_upload(
        [
            ("adapter1_bundle", "adapter1.npz", _npz(lora_B=B1, lora_A=A1)),
            ("adapter2_b", "adapter2_B.npy", _npy(B2)),
            ("adapter2_a", "adapter2_A.npy", _npy(A2)),
            ("activations", "activations.npy", _npy(X)),
        ]
    )


def _merge(client, rank, protected):
    fields = [("rank_limit", str(rank))] + [("protected", str(i)) for i in protected]
    return client.post_form("/merge", fields)


def _verify(client, activation, epsilon, pairs):
    fields = [
        ("activation_index", str(activation)),
        ("epsilon", repr(epsilon)),
    ]
    fields += [("candidate_row", str(row)) for row, _ in pairs]
    fields += [("candidate_base", repr(base)) for _, base in pairs]
    return client.post_form("/verify", fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args()

    client = Client(args.base)
    B1, A1, B2, A2, X = _fixture()
    W = B1 @ A1 + B2 @ A2

    print(f"[1] upload  {_upload(client, B1, A1, B2, A2, X)[0]}")
    print(f"[2] merge   {_merge(client, 6, [0, 1, 2])[0]}")
    status, body, headers = client.request("/download")
    print(
        f"[3] package gen={headers.get('X-Package-Generation')} "
        f"rank_limit={headers.get('X-Rank-Limit')} bytes={len(body)}"
    )
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    print(f"    manifest: generation={manifest['generation']} "
          f"rank_limit={manifest['rank_limit']} "
          f"protected={manifest['protected_activation_indices']}")

    # --- tie on the merged side: equalise base + response on activation 0 ----
    resp0 = W @ X[:, 0]
    rows = (0, 2, 4)
    tied = [(row, -float(resp0[row])) for row in rows]
    print(f"[4] verify tie  {_verify(client, 0, 1e-6, tied)[0]}")
    page = _wait(client)
    text = _text(page)
    match = re.search(r"k = (\d+)", text)
    print(f"    k={match.group(1) if match else '?'}  "
          f"missed-list={'k − 1' in text}")
    if "k − 1" in text:
        segment = text[text.find("k − 1"):].split("B 为 out")[0]
        print(f"    {segment.strip()}")

    # --- clear winner --------------------------------------------------------
    clear = [
        (1, 100.0 - float(resp0[1])),
        (3, -10.0 - float(resp0[3])),
        (5, -20.0 - float(resp0[5])),
    ]
    print(f"[5] verify clear {_verify(client, 0, 0, clear)[0]}")
    text = _text(_wait(client))
    match = re.search(r"k = (\d+)", text)
    print(f"    k={match.group(1) if match else '?'}  "
          f"missed-list={'k − 1' in text}")

    # --- re-upload withdraws the old conclusion ------------------------------
    _upload(client, B1, A1, B2, A2, X + 0.25)
    text = _text(client.page()[1])
    print(f"[6] after re-upload: old k shown={'k = ' in text}  "
          f"stale={('输入已变更，尚未重算' in text)}")
    status, _body, _headers = _verify(client, 0, 0, tied)
    text = _text(_wait(client) if status == 303 else client.page()[1])
    print(f"    verify on stale package -> {status}, "
          f"reason shown={'输入已变更、尚未重算' in text}")

    # --- recompute, then verify the new package ------------------------------
    _merge(client, 4, [0])
    status, body, headers = client.request("/download")
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        new_manifest = json.loads(zf.read("manifest.json"))
    print(f"[7] repackage gen={headers.get('X-Package-Generation')} "
          f"(manifest generation={new_manifest['generation']})")
    # Activation 0 is protected in the new package, so its merged response
    # equals the original exactly: equalising base + response still ties it.
    resp0_new = W @ (X + 0.25)[:, 0]
    tied_new = [(row, -float(resp0_new[row])) for row in rows]
    _verify(client, 0, 1e-6, tied_new)
    text = _text(_wait(client))
    match = re.search(r"k = (\d+)", text)
    version = re.search(r"包版本：第 (\d+) 版", text)
    missed = re.search(r"输出行 ([\d、]+)", text[text.find("k − 1"):]) if "k − 1" in text else None
    print(f"    k={match.group(1) if match else '?'}  "
          f"package version shown={version.group(1) if version else '?'}  "
          f"missed rows={missed.group(1) if missed else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
