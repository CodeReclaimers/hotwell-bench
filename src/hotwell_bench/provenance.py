"""Environment provenance and the gate that every generation run must pass.

A released item set is only reproducible against the exact simulator commit and
numpy version it was generated with (hotwell HANDOFF D21: numpy's
Generator.normal() is not promised stable across releases). `check_environment()`
refuses to proceed on anything else, so a drifted environment cannot silently
produce a different item set under the same seeds.
"""
from __future__ import annotations

import json
from importlib.metadata import distribution, version

#: commit of hotwell tag v0.4.0 (`git ls-remote origin refs/tags/v0.4.0^{}`)
PINNED_HOTWELL_TAG = "v0.4.0"
PINNED_HOTWELL_COMMIT = "5f576365b598e51598ced1e2c153f9e71211f33b"
PINNED_NUMPY = "2.5.1"


def hotwell_provenance() -> dict:
    """Where the installed hotwell came from, per its PEP 610 direct_url.json.
    Keys: version, url, requested_revision, commit_id, editable."""
    dist = distribution("hotwell")
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    vcs = info.get("vcs_info", {})
    return {
        "version": dist.version,
        "url": info.get("url"),
        "requested_revision": vcs.get("requested_revision"),
        "commit_id": vcs.get("commit_id"),
        "editable": bool(info.get("dir_info", {}).get("editable", False)),
    }


def environment_record() -> dict:
    """The provenance block recorded with every generated item set."""
    return {"hotwell": hotwell_provenance(), "numpy": version("numpy")}


def check_environment() -> dict:
    """Raise RuntimeError unless hotwell is the pinned, non-editable tag commit and
    numpy is the pinned version. Returns the environment record on success."""
    rec = environment_record()
    hw = rec["hotwell"]
    problems = []
    if hw["editable"]:
        problems.append("hotwell is an editable install (it must come from the pinned tag)")
    if hw["commit_id"] != PINNED_HOTWELL_COMMIT:
        problems.append(f"hotwell commit {hw['commit_id']!r} != pinned {PINNED_HOTWELL_COMMIT}"
                        f" ({PINNED_HOTWELL_TAG})")
    if rec["numpy"] != PINNED_NUMPY:
        problems.append(f"numpy {rec['numpy']} != pinned {PINNED_NUMPY}")
    if problems:
        raise RuntimeError("environment does not match the pins: " + "; ".join(problems))
    return rec
