"""Test-split secrecy guard (PLAN-bench-0.1.md §1).

The repository is public. The test split's runs, items and answer key must never
be written anywhere git could pick them up. Every writer calls
`check_output_path(path, split)` before writing. For split "test" it refuses any
path inside this package's repository or inside the git work tree containing
the current directory.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_toplevel(start: Path) -> Path | None:
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    return None


def protected_roots() -> list[Path]:
    roots = []
    if (REPO_ROOT / ".git").exists():
        roots.append(REPO_ROOT)
    cwd_top = _git_toplevel(Path.cwd().resolve())
    if cwd_top is not None and cwd_top not in roots:
        roots.append(cwd_top)
    return roots


def check_output_path(path: str | Path, split: str) -> Path:
    """Return the resolved path, or raise PermissionError for test-split output
    inside a git work tree."""
    p = Path(path).resolve()
    if split == "test":
        for root in protected_roots():
            if p == root or root in p.parents:
                raise PermissionError(
                    f"refusing to write test-split material inside git work tree {root}: {p}")
    return p
