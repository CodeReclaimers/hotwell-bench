"""Grader (B5; PLAN-bench-0.1.md §6, B-Q2).

Per item, from its key (keys/<id>.json) and an answer object:
- detection: `fault_present` equals the key's;
- category: for a fault item, the answer's category is the category of one of the
  accepted answers (the true fault or a confusable one). For a null item it is `none`;
- component: the (category, component) pair is one of the accepted answers.
  Comparison is case-insensitive and ignores surrounding whitespace. A plant-wide
  category takes null (also accepted: "", "-", "—");
- onset: fault items only. The onset lies in the key's window [t_inj − 600 s,
  t_det + 600 s]. For null items onset is not scored;
- full: null item → detection and category; fault item → detection, component
  (which implies category) and onset;
- Brier: (confidence − full)², over valid answers.

An answer that fails validation (not an object, missing or mistyped field, category
outside the closed list, unparseable onset, confidence outside [0, 1]) scores 0 on
every criterion and is counted as invalid. It is excluded from the Brier mean. A
missing answer is invalid.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from hotwell_bench.taxonomy import CATEGORIES, FAULT_CATEGORIES, NULL_CATEGORY, plantwide

_ONSET = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?\s*$")
_NULLISH = {None, "", "-", "—", "null", "none"}
CRITERIA = ("detection", "category", "component", "onset", "full")


def onset_seconds(s: str) -> int:
    m = _ONSET.match(s)
    if not m:
        raise ValueError(f"onset {s!r} is not HH:MM:SS")
    h, mi, se = (int(x) for x in m.groups())
    if h > 24 or mi > 59 or se > 59:
        raise ValueError(f"onset {s!r} out of range")
    return h * 3600 + mi * 60 + se


def _norm(c):
    if isinstance(c, str):
        c = c.strip()
        if c.lower() in _NULLISH:
            return None
        return c.lower()
    return None if c is None else c


def validate(ans) -> tuple[dict | None, str | None]:
    """(normalized answer, None) or (None, reason)."""
    if not isinstance(ans, dict):
        return None, "answer is not a JSON object"
    for k in ("fault_present", "category", "component", "onset", "confidence"):
        if k not in ans:
            return None, f"missing field {k!r}"
    if not isinstance(ans["fault_present"], bool):
        return None, "fault_present is not a boolean"
    cat = ans["category"]
    if not isinstance(cat, str) or cat.strip() not in (*FAULT_CATEGORIES, NULL_CATEGORY):
        return None, f"category {cat!r} not in the closed list"
    comp = ans["component"]
    if comp is not None and not isinstance(comp, str):
        return None, "component is neither a string nor null"
    onset = ans["onset"]
    t = None
    if onset is not None:
        if not isinstance(onset, str):
            return None, "onset is neither a string nor null"
        try:
            t = onset_seconds(onset)
        except ValueError as e:
            return None, str(e)
    conf = ans["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (
            math.isfinite(conf) and 0.0 <= conf <= 1.0):
        return None, f"confidence {conf!r} not a number in [0, 1]"
    return {"fault_present": ans["fault_present"], "category": cat.strip(),
            "component": _norm(comp), "onset_s": t, "confidence": float(conf)}, None


def _pair_ok(cat: str, comp, accepted: list[dict]) -> bool:
    for a in accepted:
        if a["category"] != cat:
            continue
        want = _norm(a["component"])
        if cat == NULL_CATEGORY or plantwide(cat):
            if comp is None:
                return True
        elif comp is not None and comp == want:
            return True
    return False


def grade_item(key: dict, ans) -> dict:
    a, err = validate(ans)
    out = {"item_id": key["item_id"], "fault_present": key["fault_present"], "valid": a is not None,
           "error": err}
    if a is None:
        out.update({c: False for c in CRITERIA})
        out["brier"] = None
        return out
    det = a["fault_present"] == key["fault_present"]
    if not key["fault_present"]:
        cat = a["category"] == NULL_CATEGORY
        comp = cat and a["component"] is None
        onset = None
        full = det and cat
    else:
        cats = {x["category"] for x in key["accepted_answers"]}
        cat = a["category"] in cats
        comp = cat and _pair_ok(a["category"], a["component"], key["accepted_answers"])
        lo, hi = key["onset_window"]
        onset = a["onset_s"] is not None and lo <= a["onset_s"] <= hi
        full = det and comp and onset
        if a["onset_s"] is not None:
            out["onset_err_s"] = a["onset_s"] - key["t_inj"]
            out["onset_minus_tdet_s"] = a["onset_s"] - key["t_det"]
    out.update({"detection": det, "category": cat, "component": comp, "onset": onset,
                "full": full, "confidence": a["confidence"],
                "brier": (a["confidence"] - float(full)) ** 2})
    return out


def load_keys(keys_dir: str | Path) -> dict[str, dict]:
    return {json.loads(p.read_text())["item_id"]: json.loads(p.read_text())
            for p in sorted(Path(keys_dir).glob("*.json")) if p.name != "manifest.json"}


def _ratio_bucket(r) -> str:
    if r is None:
        return "n/a"
    return "<1" if r < 1 else "1-2" if r < 2 else ">=2"


def strata(key: dict) -> dict[str, str]:
    if not key["fault_present"]:
        return {"category": NULL_CATEGORY, "ratio": "null", "shape": "null", "easy": "null",
                "sensor_kind": "null"}
    s = key["strata"]
    return {"category": s["category"], "ratio": _ratio_bucket(s["ratio"]),
            "shape": s["shape"], "easy": "easy" if key["easy"] else "not easy",
            # instrument faults only; "n/a" for every other fault category
            "sensor_kind": s.get("sensor_kind") or "n/a"}


def summarize(keys: dict[str, dict], graded: list[dict]) -> dict:
    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        d = {"n": n}
        for c in CRITERIA:
            vals = [r[c] for r in rows if r[c] is not None]
            d[c] = (sum(vals) / len(vals)) if vals else None
        br = [r["brier"] for r in rows if r["brier"] is not None]
        d["brier"] = sum(br) / len(br) if br else None
        d["invalid"] = sum(not r["valid"] for r in rows)
        return d

    out = {"overall": agg(graded),
           "fault_items": agg([g for g in graded if g["fault_present"]]),
           "null_items": agg([g for g in graded if not g["fault_present"]])}
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for g in graded:
        for dim, val in strata(keys[g["item_id"]]).items():
            by[dim][val].append(g)
    out["strata"] = {dim: {v: agg(rows) for v, rows in sorted(vals.items())}
                     for dim, vals in by.items()}
    errs = sorted(abs(g["onset_err_s"]) for g in graded if "onset_err_s" in g)
    out["onset_abs_err_s_median"] = errs[len(errs) // 2] if errs else None
    return out


def grade(keys: dict[str, dict], answers: dict[str, object]) -> tuple[list[dict], dict]:
    """answers: item_id -> parsed answer (missing ids are graded invalid)."""
    graded = [grade_item(keys[i], answers.get(i)) for i in sorted(keys)]
    return graded, summarize(keys, graded)


def report(summary: dict) -> str:
    def row(name, d):
        f = lambda x: "  -  " if x is None else f"{x:5.3f}"  # noqa: E731
        return (f"{name:34s} {d['n']:4d} " + " ".join(f(d[c]) for c in CRITERIA)
                + f" {f(d['brier'])} {d['invalid']:4d}")
    lines = [f"{'':34s} {'n':>4s} " + " ".join(f"{c[:5]:>5s}" for c in CRITERIA)
             + f" {'brier':>5s} {'inval':>4s}"]
    for k in ("overall", "fault_items", "null_items"):
        lines.append(row(k, summary[k]))
    for dim, vals in summary["strata"].items():
        lines.append(f"-- by {dim}")
        for v, d in vals.items():
            lines.append(row(f"  {v}", d))
    lines.append(f"median |onset - t_inj| [s]: {summary['onset_abs_err_s_median']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ baselines

def oracle_answer(key: dict) -> dict:
    if not key["fault_present"]:
        return {"fault_present": False, "category": NULL_CATEGORY, "component": None,
                "onset": None, "confidence": 1.0}
    a = key["accepted_answers"][0]
    t = key["t_inj"]
    return {"fault_present": True, "category": a["category"], "component": a["component"],
            "onset": f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}", "confidence": 1.0}


def always_none(_key: dict, p_null: float = 0.2) -> dict:
    return {"fault_present": False, "category": NULL_CATEGORY, "component": None,
            "onset": None, "confidence": p_null}


def prior_answer(_key: dict, category: str, component, onset: str) -> dict:
    """The single most frequent (category, component) of a reference split, at a fixed
    onset: what a guesser with only the class prior can do."""
    return {"fault_present": True, "category": category, "component": component,
            "onset": onset, "confidence": 0.0}


def answer_vocabulary() -> dict[str, list]:
    return {c: [None] if plantwide(c) else list(CATEGORIES[c][1]) for c in FAULT_CATEGORIES}
