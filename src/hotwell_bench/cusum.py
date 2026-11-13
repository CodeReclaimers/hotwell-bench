"""Non-LLM change-detector baseline (plan §6): per-feature CUSUM on load-normalized
residuals, plus the category of the nearest training signature.

Reads only what a model receives (`items/<id>/instruments.csv.gz`); keys are used for
training items only. Trained on the dev split, evaluated on any split.

Features, at 1-minute resolution (minute means and minute standard deviations of the
1 Hz record):
- the 18 instruments outside the pump pairs, as they are;
- each pump pair (CP, FP) folded into roles: running-pump current, running-pump NPSH
  margin, standby-pump NPSH margin. The running pump is the one with the larger
  current in that minute, so a swap does not look like a step. Minutes within
  `SWAP_GUARD_MIN` of a swap are not used (CUSUM holds).
For each feature: the mean and log(std + eps).

Load normalization. A load estimate L is the median of five load proxies (mw_out,
steam_flow, condensate_flow, firing_rate, throttle_pos), each mapped to MW by a linear
fit on training data; the median keeps one faulted proxy from moving L. Every feature
is regressed on [1, L, L^2, L(t)-L(t-10 min), L(t)-L(t-30 min)] over fault-free
training minutes; the residual is divided by its robust sigma (1.4826 MAD).

CUSUM. Two-sided, drift k = K_SIGMA; threshold per feature and direction is
H_MARGIN x the largest statistic reached on the fault-free training minutes (at
least H_FLOOR). Detection = first minute any feature crosses. Onset = the start of the
run that ended in that crossing (last minute the statistic was 0, plus one).

Identification. Signature = for each feature, the direction of its first crossing
within SIG_WINDOW_MIN after detection (+1, -1, 0), plus the first feature. Nearest
training fault item by sum |s - s'| + FIRST_WEIGHT x [first feature differs] gives the
category. Components: instrument_fault -> the raw instrument behind the first
feature; pump_performance_loss -> the running pump of the neighbour's pair at
detection; other categories -> the neighbour's component.
Confidence is a constant 0.5 (the detector has no calibrated probability).

Known weakness (by construction, measured): thresholds come from finite training
maxima while 96 one-sided statistics run every minute, so some unseen fault-free days
alarm. Trained on the 0.1 dev split, 8 of the 39 test null items alarmed.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hotwell_bench.taxonomy import NULL_CATEGORY

PAIRS = {"cp": ("cp1", "cp2"), "fp": ("fp1", "fp2")}
LOAD_PROXIES = ("mw_out", "steam_flow", "condensate_flow", "firing_rate", "throttle_pos")
SWAP_GUARD_MIN = 5
WARMUP_MIN = 30
K_SIGMA = 1.0
H_MARGIN = 1.25
H_FLOOR = 5.0
SIG_WINDOW_MIN = 120
FIRST_WEIGHT = 3.0
EPS_STD = 1e-9


def read_instruments(item_dir: Path) -> tuple[list[str], np.ndarray]:
    """(names without units, array [N, 26]); the t_s column is dropped."""
    lines = gzip.decompress((item_dir / "instruments.csv.gz").read_bytes()).decode().split("\n")
    hdr = [h.split(" [")[0] for h in lines[1].split(",")[1:]]
    a = np.array([r.split(",")[1:] for r in lines[2:] if r], dtype=float)
    return hdr, a


@dataclass
class Minutes:
    names: list[str]          # feature names (role names for pump pairs)
    x: np.ndarray             # [M, F] features
    running: dict[str, np.ndarray]  # pair -> [M] index (0/1) of the running pump
    usable: np.ndarray        # [M] bool, False near swaps
    load_raw: np.ndarray      # [M, len(LOAD_PROXIES)] minute means of the proxies


def minutes(hdr: list[str], a: np.ndarray) -> Minutes:
    m = a.shape[0] // 60
    blk = a[: m * 60].reshape(m, 60, a.shape[1])
    mean, std = blk.mean(1), blk.std(1)
    col = {h: i for i, h in enumerate(hdr)}
    paired = {f"{p}_current" for pr in PAIRS.values() for p in pr} | {
        f"{p}_npsh_margin" for pr in PAIRS.values() for p in pr}
    names, cols = [], []
    for h in hdr:
        if h in paired:
            continue
        names += [f"{h}:mean", f"{h}:logstd"]
        cols += [mean[:, col[h]], np.log(std[:, col[h]] + EPS_STD)]
    running, usable = {}, np.ones(m, bool)
    for pair, (a1, a2) in PAIRS.items():
        cur = np.stack([mean[:, col[f"{a1}_current"]], mean[:, col[f"{a2}_current"]]], 1)
        run = np.argmax(cur, 1)
        running[pair] = run
        rows = np.arange(m)
        for role, idx in (("run", run), ("sb", 1 - run)):
            for q in (("current",) if role == "run" else ()) + ("npsh_margin",):
                c = np.stack([col[f"{a1}_{q}"], col[f"{a2}_{q}"]])[idx]
                names += [f"{pair}_{role}_{q}:mean", f"{pair}_{role}_{q}:logstd"]
                cols += [mean[rows, c], np.log(std[rows, c] + EPS_STD)]
        for s in np.flatnonzero(np.diff(run)) + 1:
            usable[max(0, s - SWAP_GUARD_MIN): s + SWAP_GUARD_MIN] = False
    load_raw = np.stack([mean[:, col[p]] for p in LOAD_PROXIES], 1)
    return Minutes(names, np.stack(cols, 1), running, usable, load_raw)


def _design(L: np.ndarray) -> np.ndarray:
    d10 = np.r_[np.zeros(10), L[10:] - L[:-10]]
    d30 = np.r_[np.zeros(30), L[30:] - L[:-30]]
    return np.stack([np.ones_like(L), L, L * L, d10, d30], 1)


def load_estimate(mn: Minutes, proxy_fit: np.ndarray) -> np.ndarray:
    """proxy_fit[j] = (a, b) with proxy_j ~ a + b * mw_out[MW]; inverted per proxy."""
    est = (mn.load_raw - proxy_fit[:, 0]) / proxy_fit[:, 1]
    return np.median(est, 1)


def _cusum(z: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    hi, lo = np.zeros_like(z), np.zeros_like(z)
    sh = sl = np.zeros(z.shape[1])
    for t in range(z.shape[0]):
        zt = np.nan_to_num(z[t])          # NaN (unusable minute): hold the statistic
        sh = np.where(np.isnan(z[t]), sh, np.maximum(0.0, sh + zt - k))
        sl = np.where(np.isnan(z[t]), sl, np.maximum(0.0, sl - zt - k))
        hi[t], lo[t] = sh, sl
    return hi, lo


def residual_z(mn: Minutes, model: dict) -> np.ndarray:
    L = load_estimate(mn, np.array(model["proxy_fit"]))
    r = (mn.x - _design(L) @ np.array(model["beta"])) / np.array(model["sigma"])
    r[~mn.usable] = np.nan
    r[:WARMUP_MIN] = np.nan
    return r


def fit(train: list[tuple[Minutes, int]]) -> dict:
    """train: (minutes, number of fault-free leading minutes) per training item."""
    names = train[0][0].names
    lr = np.concatenate([mn.load_raw[:n][mn.usable[:n]] for mn, n in train])
    mw = lr[:, 0] / 1e6
    proxy_fit = np.array([np.polyfit(mw, lr[:, j], 1)[::-1] for j in range(lr.shape[1])])
    proxy_fit[0] = [0.0, 1e6]          # mw_out itself, W -> MW
    X, Y = [], []
    for mn, n in train:
        L = load_estimate(mn, proxy_fit)
        ok = mn.usable[:n].copy()
        ok[:WARMUP_MIN] = False
        X.append(_design(L)[:n][ok])
        Y.append(mn.x[:n][ok])
    X, Y = np.concatenate(X), np.concatenate(Y)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    sigma = 1.4826 * np.median(np.abs(res - np.median(res, 0)), 0)
    sigma = np.where(sigma > 0, sigma, res.std(0) + 1e-12)
    model = {"names": names, "proxy_fit": proxy_fit.tolist(), "beta": beta.tolist(),
             "sigma": sigma.tolist(), "k": K_SIGMA}
    hmax = np.zeros((2, len(names)))
    for mn, n in train:
        hi, lo = _cusum(residual_z(mn, model)[:n], K_SIGMA)
        if n:
            hmax = np.maximum(hmax, np.stack([hi.max(0), lo.max(0)]))
    model["h"] = np.maximum(H_MARGIN * hmax, H_FLOOR).tolist()
    return model


@dataclass
class Detection:
    t_alarm_min: int | None
    onset_min: int | None
    first: int | None               # feature index
    signature: list[int]            # per feature: +1, -1, 0


def detect(mn: Minutes, model: dict) -> Detection:
    z = residual_z(mn, model)
    hi, lo = _cusum(z, model["k"])
    h = np.array(model["h"])
    over = np.stack([hi > h[0], lo > h[1]])          # [2, M, F]
    any_t = over.any((0, 2))
    if not any_t.any():
        return Detection(None, None, None, [0] * len(model["names"]))
    t = int(np.argmax(any_t))
    d, f = (int(v[0]) for v in np.nonzero(over[:, t, :]))
    stat = (hi if d == 0 else lo)[: t + 1, f]
    zeros = np.flatnonzero(stat == 0)
    onset = int(zeros[-1] + 1) if zeros.size else 0
    sig = []
    end = min(over.shape[1], t + SIG_WINDOW_MIN)
    for j in range(over.shape[2]):
        up, dn = over[0, t:end, j], over[1, t:end, j]
        tu = np.argmax(up) if up.any() else None
        td = np.argmax(dn) if dn.any() else None
        if tu is None and td is None:
            sig.append(0)
        elif td is None or (tu is not None and tu <= td):
            sig.append(1)
        else:
            sig.append(-1)
    return Detection(t, onset, f, sig)


def raw_instrument(feature: str, mn: Minutes, t: int) -> str:
    """Feature name -> the instrument name the answer vocabulary uses."""
    base = feature.split(":")[0]
    for pair, (a1, a2) in PAIRS.items():
        if base.startswith(f"{pair}_"):
            _, role, *q = base.split("_")
            run = int(mn.running[pair][t])
            phys = (a1, a2)[run if role == "run" else 1 - run]
            return f"{phys}_{'_'.join(q)}"
    return base


def _pump(pair_letters: str, mn: Minutes, t: int) -> str:
    pair = pair_letters.lower()
    return f"{pair_letters.upper()}-{int(mn.running[pair][t]) + 1}"


def onset_hms(minute: int) -> str:
    s = minute * 60
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:00"


def answer(mn: Minutes, model: dict, train_sigs: list[dict]) -> tuple[dict, Detection]:
    det = detect(mn, model)
    if det.t_alarm_min is None or not train_sigs:
        return {"fault_present": False, "category": NULL_CATEGORY, "component": None,
                "onset": None, "confidence": 0.5}, det
    s = np.array(det.signature)
    best = min(train_sigs, key=lambda r: (
        np.abs(s - np.array(r["signature"])).sum()
        + FIRST_WEIGHT * (r["first"] != det.first), r["item_id"]))
    cat, comp = best["category"], best["component"]
    if cat == "instrument_fault":
        comp = raw_instrument(model["names"][det.first], mn, det.t_alarm_min)
    elif cat == "pump_performance_loss" and comp:
        comp = _pump(comp.split("-")[0], mn, det.t_alarm_min)
    return {"fault_present": True, "category": cat, "component": comp,
            "onset": onset_hms(det.onset_min), "confidence": 0.5}, det


def train_from_split(split_dir: Path) -> tuple[dict, list[dict]]:
    """Fit on a split with keys (dev): fault-free minutes from null items (whole record)
    and fault items (before t_inj); signatures from fault items the model detects."""
    keys = {p.stem: json.loads(p.read_text()) for p in sorted((split_dir / "keys").glob("*.json"))
            if p.name != "manifest.json"}
    data = {}
    for i, k in keys.items():
        mn = minutes(*read_instruments(split_dir / "items" / i))
        clean = mn.x.shape[0] if not k["fault_present"] else max(0, k["t_inj"] // 60 - 1)
        data[i] = (mn, clean, k)
    model = fit([(mn, n) for mn, n, _ in data.values()])
    sigs = []
    for i, (mn, _, k) in data.items():
        if not k["fault_present"]:
            continue
        det = detect(mn, model)
        if det.t_alarm_min is None:
            continue
        a = k["accepted_answers"][0]
        sigs.append({"item_id": i, "category": a["category"], "component": a["component"],
                     "signature": det.signature, "first": det.first})
    return model, sigs
