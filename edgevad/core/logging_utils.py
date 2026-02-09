"""Logging, timestamps, environment info."""

from __future__ import annotations

import datetime as _dt
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def utc_ts() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_git_hash(repo_dir: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", out):
            return out
    except Exception:
        pass
    return None


def setup_logger(name: str = "edgevad", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def env_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = sys.version.replace("\n", " ")
    info["platform"] = platform.platform()
    info["torch"] = torch.__version__

    try:
        import torchvision
        info["torchvision"] = getattr(torchvision, "__version__", None)
    except Exception:
        info["torchvision"] = None

    try:
        import ultralytics
        info["ultralytics"] = getattr(ultralytics, "__version__", None)
    except Exception:
        info["ultralytics"] = None

    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available() and "cuda" in str(device):
        try:
            idx = int(str(device).split(":")[-1])
        except Exception:
            idx = 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
    else:
        info["gpu_name"] = None
    return info
