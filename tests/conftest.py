import sys
from pathlib import Path


def pytest_configure():
    repo_root = Path(__file__).resolve().parents[1]
    edgevad_dir = repo_root / "edgevad"
    for p in (repo_root, edgevad_dir):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
