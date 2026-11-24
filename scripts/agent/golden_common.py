"""Shared helpers for the golden-fixture and benchmark scripts (task A0).

Used by ``make_golden.py``, ``compare_golden.py`` and ``benchmark.py`` in this
directory. Nothing here imports ``lpdm``: the scripts drive the model as a
subprocess (``python -m lpdm.main``), exactly as a user would, and read its
documented outputs (``footprints.zarr``, ``endpoint_particles.parquet``,
``run_metadata.json``).

Fixture layout (one directory per config and device)::

    <fixture>/manifest.json            provenance and checksums (always committed)
    <data>/footprint.npz               footprint [release, time_ago, z_bin, lat, lon]
                                       float32, plus release_time, compressed
    <data>/endpoint_particles.parquet  copied from the run
    <data>/run_metadata.json           copied from the run

``<data>`` is the fixture directory itself for small fixtures, or a directory
under the outputs directory for large ones (the manifest's ``data_location``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_ROOT = REPO_ROOT / "tests" / "golden"
TIME_BIN = Path("/usr/bin/time")

FOOTPRINT_FILE = "footprint.npz"
ENDPOINT_FILE = "endpoint_particles.parquet"
METADATA_FILE = "run_metadata.json"
MANIFEST_FILE = "manifest.json"


# --------------------------------------------------------------------------- paths


def main_checkout_root() -> Path:
    """The main checkout of this repository (differs from REPO_ROOT in a git worktree)."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return Path(common).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        return REPO_ROOT


def outputs_root() -> Path:
    """``$GLIDE_AGENT_OUTPUTS`` if set, else ``outputs/`` of the main checkout."""
    env = os.environ.get("GLIDE_AGENT_OUTPUTS")
    return Path(env) if env else main_checkout_root() / "outputs"


def portable_location(path: Path) -> str:
    """Describe ``path`` without writing machine-specific values into tracked files.

    Under ``$GLIDE_AGENT_OUTPUTS`` → ``$GLIDE_AGENT_OUTPUTS/...``; under the main
    checkout → a repo-relative path; otherwise the absolute path.
    """
    path = path.resolve()
    env = os.environ.get("GLIDE_AGENT_OUTPUTS")
    if env:
        try:
            return "$GLIDE_AGENT_OUTPUTS/" + str(path.relative_to(Path(env).resolve()))
        except ValueError:
            pass
    for root in (REPO_ROOT, main_checkout_root()):
        try:
            return str(path.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(path)


def resolve_location(location: str) -> Path:
    """Inverse of :func:`portable_location`."""
    if location.startswith("$GLIDE_AGENT_OUTPUTS/"):
        env = os.environ.get("GLIDE_AGENT_OUTPUTS")
        rest = location.split("/", 1)[1]
        if env:
            return Path(env) / rest
        return main_checkout_root() / "outputs" / rest
    p = Path(location)
    if p.is_absolute():
        return p
    for root in (REPO_ROOT, main_checkout_root()):
        if (root / p).exists():
            return root / p
    return main_checkout_root() / p


def fixture_data_dir(fixture_dir: Path, manifest: dict, override: Path | None = None) -> Path:
    if override is not None:
        return override
    loc = manifest.get("data_location", ".")
    if loc == ".":
        return fixture_dir
    return resolve_location(loc)


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hashes(config_path: Path, resolved_config: dict | None) -> dict[str, str]:
    """SHA-256 of the YAML file and of the resolved config (minus run-specific keys).

    The resolved hash ignores ``io`` (paths) and ``simulation.device`` (set on the
    command line), so a fixture and a fresh run on the same rev hash the same.
    """
    out = {"config_file_sha256": sha256_file(config_path)}
    if resolved_config is not None:
        cfg = json.loads(json.dumps(resolved_config))
        cfg.pop("io", None)
        cfg.get("simulation", {}).pop("device", None)
        blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
        out["resolved_config_sha256"] = hashlib.sha256(blob).hexdigest()
    return out


# --------------------------------------------------------------------------- provenance


def git_info() -> dict[str, object]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        rev = _git("rev-parse", "HEAD")
        dirty_src = bool(_git("status", "--porcelain", "--", "src"))
        return {"git_rev": rev, "src_dirty": dirty_src}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_rev": "unknown", "src_dirty": None}


def torch_info(gpu: bool = True) -> dict[str, object]:
    import torch

    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }
    if gpu and torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def met_provenance(run_metadata: dict) -> list[dict]:
    """The met stores' provenance from ``run_metadata.json`` (recipe text dropped)."""
    keep = (
        "domain",
        "producer",
        "producer_version",
        "producer_commit",
        "recipe",
        "glide_format",
        "glide_format_version",
        "glide_vertical_coordinate",
        "compression",
        "processing",
    )
    stores = run_metadata.get("runtime", {}).get("met_stores", []) or []
    return [{k: s[k] for k in keep if k in s} for s in stores]


# --------------------------------------------------------------------------- running


@dataclass
class RunRecord:
    output_dir: Path
    log_path: Path
    returncode: int
    wall_seconds: float
    max_rss_kb: int | None = None
    time_v: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


RECORDED_ENV = (
    "GLIDE_COMPILE",
    "GLIDE_PHASE_TIMERS",
    "GLIDE_PHASE_TIMERS_SYNC",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
)


def run_glide(
    config: Path,
    device: str,
    output_dir: Path,
    *,
    phase_timers: bool = False,
    extra_env: dict[str, str] | None = None,
) -> RunRecord:
    """Run ``python -m lpdm.main`` (under ``/usr/bin/time -v`` when available).

    stdout and stderr go to ``<output_dir>.log`` (next to the output directory,
    so the model's own outputs are untouched). Raises on a non-zero exit.
    """
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory exists, refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.with_name(output_dir.name + ".log")
    env = dict(os.environ)
    # Run the model from this checkout's source (the rev the manifest records),
    # even when the venv's editable install points at another checkout.
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if phase_timers:
        env["GLIDE_PHASE_TIMERS"] = "1"
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable,
        "-m",
        "lpdm.main",
        "--config",
        str(config),
        "--device",
        device,
        "--output-uri",
        str(output_dir),
    ]
    if TIME_BIN.exists():
        cmd = [str(TIME_BIN), "-v", *cmd]
    print(f"running: {' '.join(cmd)}\n  log: {log_path}", flush=True)
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.monotonic() - t0
    text = log_path.read_text(errors="replace")
    tv = parse_time_v(text)
    rec = RunRecord(
        output_dir=output_dir,
        log_path=log_path,
        returncode=proc.returncode,
        wall_seconds=wall,
        max_rss_kb=int(tv["Maximum resident set size (kbytes)"])
        if "Maximum resident set size (kbytes)" in tv
        else None,
        time_v=tv,
        env={k: env[k] for k in RECORDED_ENV if k in env},
    )
    if proc.returncode != 0:
        raise SystemExit(f"lpdm.main failed (exit {proc.returncode}); see {log_path}")
    print(
        f"  done in {wall:.1f} s"
        + (f", peak RSS {rec.max_rss_kb / 1e6:.1f} GB" if rec.max_rss_kb else ""),
        flush=True,
    )
    return rec


_TIME_V_LINE = re.compile(r"^\t(.+?): (.+)$")


def parse_time_v(text: str) -> dict[str, str]:
    """Parse the block GNU ``time -v`` appends to a log."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _TIME_V_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# --------------------------------------------------------------------------- outputs


def read_run_footprint(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Dense footprint [release, time_ago, z_bin, lat, lon] and release times (int64)."""
    import zarr

    g = zarr.open_group(str(output_dir / "footprints.zarr"), mode="r")
    return np.asarray(g["footprint"][...]), np.asarray(g["release_time"][...])


def read_fixture_footprint(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(data_dir / FOOTPRINT_FILE) as z:
        return z["footprint"], z["release_time"]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
