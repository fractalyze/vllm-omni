# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The ``/v1/videos`` driver: one request, measured the way the plan defines it.

The showcase's headline metric is **request wall time, from the POST to
``/v1/videos`` until the MP4 is downloadable, on a warm server**. That is what
:func:`submit_and_fetch` returns, and the definition fixes three things the
obvious implementation gets wrong:

- The clock starts before the POST and stops after the content ``GET``
  returns the last byte, so the download is inside the measurement. A number
  that stops at ``status == completed`` is not this metric.
- Polling is a cost the client pays, not the server: a 2 s poll interval
  would quantize a 40 s request into 2 s steps and inflate every number by
  half an interval on average. The poll interval is therefore small
  (:data:`POLL_INTERVAL_S`) and reported, so the quantization it can
  contribute is visible beside the number.
- ``/v1/videos`` takes multipart form fields, not a JSON body (see the
  Kandinsky 6 recipe). Sending JSON gets a 400 that looks like a model
  failure.

Nothing here imports vLLM-Omni: the driver talks to a server over HTTP and is
deliberately installable in a bare venv, so Track C can drive a server built
from a different commit than the harness.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POLL_INTERVAL_S = 0.05
DEFAULT_TIMEOUT_S = 3600.0


class RequestFailedError(RuntimeError):
    """The server reported a failed generation, or never reached ``completed``."""


@dataclass
class RequestResult:
    """One ``/v1/videos`` request, with the timings the plan's metrics need."""

    request_wall_s: float
    """POST sent -> MP4 fully downloaded. The headline metric."""
    submit_s: float
    """POST sent -> the server returned a job id."""
    generate_s: float
    """Job id -> the first poll that saw ``completed``."""
    download_s: float
    """``completed`` -> the last byte of the MP4."""
    video_id: str
    mp4_path: Path
    mp4_bytes: int
    n_polls: int
    poll_interval_s: float
    status_payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "mp4_path"}
        out["mp4_path"] = str(self.mp4_path)
        return out


def _post_multipart(url: str, fields: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST ``fields`` as ``multipart/form-data`` and decode the JSON reply."""
    boundary = f"----k6bench{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:  # surface the server's own message
        raise RequestFailedError(f"POST {url} -> {exc.code}: {exc.read().decode()[:2000]}") from exc


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def request_fields(
    prompt: str,
    *,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    guidance_scale: float | None = None,
    negative_prompt: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, str]:
    """The form fields for one T2VA request.

    ``size`` is ``WIDTHxHEIGHT``, matching the recipe's ``-F size=864x480``.
    """
    fields: dict[str, str] = {
        "prompt": prompt,
        "size": f"{width}x{height}",
        "num_frames": str(num_frames),
        "num_inference_steps": str(num_inference_steps),
        "seed": str(seed),
    }
    if guidance_scale is not None:
        fields["guidance_scale"] = str(guidance_scale)
    if negative_prompt is not None:
        fields["negative_prompt"] = negative_prompt
    if extra_body:
        fields["extra_body"] = json.dumps(extra_body)
    return fields


def submit_and_fetch(
    base_url: str,
    fields: dict[str, str],
    out_path: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> RequestResult:
    """Run one request end to end and write the MP4 to ``out_path``.

    Raises :class:`RequestFailedError` if the server reports ``failed`` or the job
    has not completed within ``timeout_s``; a timed-out request is a failed
    measurement, never a slow one to be recorded.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = base_url.rstrip("/")

    t_start = time.perf_counter()
    created = _post_multipart(f"{base_url}/v1/videos", fields, timeout=timeout_s)
    t_submitted = time.perf_counter()

    video_id = created.get("id")
    if not video_id:
        raise RequestFailedError(f"POST /v1/videos returned no id: {created!r}")

    deadline = t_start + timeout_s
    n_polls = 0
    payload: dict[str, Any] = created
    # A server that answers the POST with a terminal status needs no polling.
    while payload.get("status") not in {"completed", "failed"}:
        if time.perf_counter() > deadline:
            raise RequestFailedError(f"video {video_id} still {payload.get('status')!r} after {timeout_s}s")
        time.sleep(poll_interval_s)
        n_polls += 1
        payload = _get_json(f"{base_url}/v1/videos/{video_id}", timeout=60.0)
    if payload.get("status") == "failed":
        raise RequestFailedError(f"video {video_id} failed: {payload.get('error')!r}")
    t_completed = time.perf_counter()

    with urllib.request.urlopen(f"{base_url}/v1/videos/{video_id}/content", timeout=timeout_s) as response:
        data = response.read()
    out_path.write_bytes(data)
    t_done = time.perf_counter()

    if not data:
        raise RequestFailedError(f"video {video_id} returned an empty body")

    return RequestResult(
        request_wall_s=t_done - t_start,
        submit_s=t_submitted - t_start,
        generate_s=t_completed - t_submitted,
        download_s=t_done - t_completed,
        video_id=str(video_id),
        mp4_path=out_path,
        mp4_bytes=len(data),
        n_polls=n_polls,
        poll_interval_s=poll_interval_s,
        status_payload=payload,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="One /v1/videos request, timed.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = submit_and_fetch(
        args.base_url,
        request_fields(
            args.prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance,
        ),
        args.out,
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
