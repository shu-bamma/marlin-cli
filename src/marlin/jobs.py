"""Async jobs as files — one invariant for local and hosted backends.

`marlin index --async` detaches a worker and returns a job id immediately;
`marlin status <id>` reads ~/.marlin/jobs/<id>.json. Same contract the
SKILL.md teaches agents, regardless of where inference runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import JOBS_DIR


def _job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def write_job(job_id: str, data: dict) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = _job_file(job_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_job_file(job_id))


def read_job(job_id: str) -> dict | None:
    f = _job_file(job_id)
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def list_jobs() -> list[dict]:
    if not JOBS_DIR.is_dir():
        return []
    out = []
    for f in sorted(JOBS_DIR.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def spawn_index(inputs: list[str], *, stt: bool) -> str:
    job_id = uuid.uuid4().hex[:8]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log = (JOBS_DIR / f"{job_id}.log").open("w")
    cmd = [sys.executable, "-m", "marlin.cli", "index", *inputs, "--job", job_id]
    if stt:
        cmd.append("--stt")
    subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=os.environ.copy(),
    )
    write_job(job_id, {"job_id": job_id, "state": "started", "inputs": inputs})
    return job_id
