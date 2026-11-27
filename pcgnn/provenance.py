"""Record what produced a run: code version, environment and hardware."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

_PACKAGES = ["torch", "torch_geometric", "numpy", "scikit-learn", "scipy", "pandas",
             "sentence-transformers", "codecarbon", "rdflib", "xgboost"]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:  # not a git checkout, or git missing
        return ""


def collect() -> dict:
    versions = {}
    for p in _PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    info = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [l for l in _git("status", "--porcelain").splitlines() if l],
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def write(path: Path, extra: dict | None = None) -> None:
    info = collect()
    if extra:
        info.update(extra)
    diff = _git("diff", "HEAD")
    if diff:
        (path.parent / "git_diff.patch").write_text(diff + "\n")
        info["git"]["diff_file"] = "git_diff.patch"
    path.write_text(json.dumps(info, indent=2))
