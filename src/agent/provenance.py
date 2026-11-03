"""Which code produced a run: git HEAD and whether the working tree had changes."""

import subprocess
from collections.abc import Callable

from agent.config import REPO_ROOT

GitRunner = Callable[[list[str]], str]


def run_git(args: list[str]) -> str:
    """Run a local git command in the repo and return stdout (no network)."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout


def code_version(git: GitRunner = run_git) -> dict | None:
    """{"commit": short HEAD hash, "dirty": bool}, or None if git is unavailable."""
    try:
        commit = git(["rev-parse", "--short", "HEAD"]).strip()
        dirty = bool(git(["status", "--porcelain"]).strip())
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty} if commit else None
