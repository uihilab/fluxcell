"""
Comparison of the NTHMP benchmark runs (tools/nthmp, data/figures/nthmp-bp6.json and nthmp-bp7.json)
with the laboratory records in data/nthmp (BP06: Briggs et al., 1995; BP07: Liu et al., 2008), and
the summary written to docs/results/. Every quantity below is computed from those files; the only
alignment is a single time shift per BP6 run, fixed by the peak of the incident wave at gauge 1.

Run: python tools/nthmp/analysis.py
"""
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repository root
LAB6 = ROOT / "data" / "nthmp" / "BP06"
LAB7 = ROOT / "data" / "nthmp" / "BP07"
FIELDS = ROOT / "data" / "figures"
GAUGES6 = ["1", "2", "3", "4", "6", "9", "16", "22"]
LAB_FILES = {"A": "ts2a.txt", "B": "ts2b.txt", "C": "ts2cnew1.txt"}
RUNUP_FILES = {"A": "run2a.txt", "B": "run2b.txt", "C": "run2c.txt"}
DEPTH = 0.32


def decode(text, shape=None):
    a = np.frombuffer(base64.b64decode(text), dtype=np.float32)
    return a.reshape(shape) if shape else a


def numeric_rows(path, columns):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        if len(values) >= columns:
            rows.append(values[:columns])
    return np.array(rows)


def lab_gauges(kind):
    """Laboratory water level (m) at gauges 1-4, 6, 9, 16, 22 (columns in that order; headers give m)."""
    a = numeric_rows(LAB6 / LAB_FILES[kind], 9)
    return a[:, 0], {g: a[:, k + 1] for k, g in enumerate(GAUGES6)}


def lab_runup(kind):
    """Angle (degrees) and maximum vertical runup (m; files give cm) around the island."""
    a = numeric_rows(LAB6 / RUNUP_FILES[kind], 4)
    return a[:, 1], a[:, 2] / 100.0


def peak(t, v, window=None):
    m = np.ones_like(t, bool) if window is None else (t >= window[0]) & (t <= window[1])
    i = np.flatnonzero(m)[np.argmax(v[m])]
    if 0 < i < len(v) - 1:  # parabolic refinement of the sampled maximum
        a, b, c = v[i - 1], v[i], v[i + 1]
        d = a - 2 * b + c
        if d < 0:
            s = 0.5 * (a - c) / d
            return t[i] + s * (t[1] - t[0]), b - 0.25 * (a - c) * s
    return t[i], v[i]


def bp6(run):
    kind = run["kind"]
    tl, lab = lab_gauges(kind)
    tm = np.array(run["times"])
    model = {g: np.array(run["series"][g]) for g in GAUGES6}
    # Single time shift: the incident peak at gauge 1 (lab baseline removed for the peak only).
    t1_lab, _ = peak(tl, lab["1"] - np.median(lab["1"][:50]))
    t1_mod, _ = peak(tm, model["1"])
    shift = t1_lab - t1_mod
    rows = {}
    for g in GAUGES6:
        base = np.median(lab[g][:50])
        tp_l, ap_l = peak(tl, lab[g] - base, (t1_lab - 1, t1_lab + 12))
        tp_m, ap_m = peak(tm + shift, model[g], (t1_lab - 1, t1_lab + 12))
        # Normalized RMS difference over the first 10 s after the incident peak passes gauge 1.
        window = (tl >= t1_lab - 1) & (tl <= t1_lab + 10)
        mi = np.interp(tl[window], tm + shift, model[g])
        nrmsd = float(np.sqrt(np.mean((mi - (lab[g][window] - base)) ** 2)) / ap_l)
        rows[g] = {"labPeak_m": float(ap_l), "modelPeak_m": float(ap_m), "peakError": float((ap_m - ap_l) / ap_l),
                   "labPeakTime_s": float(tp_l), "modelPeakTime_s": float(tp_m), "peakTimeError_s": float(tp_m - tp_l),
                   "nrmsd": nrmsd, "labBaseline_m": float(base)}
    angles, measured = lab_runup(kind)
    result = {"kind": kind, "measuredIncident": run.get("measured", False), "H_m": run["H"], "dx": run["dx"], "manning": run["manning"], "cells": run["width"] * run["height"],
              "retries": run["counters"]["retries"], "timeShift_s": float(shift), "gauges": rows, "runup": {}}
    grid = np.arange(0, 360, 0.5)
    for threshold, values in run["runup"].items():
        model_runup = np.interp(angles % 360, grid, np.array(values, float), period=360)
        rel = (model_runup - measured) / measured
        back = np.argmin(np.abs(angles - 90.0))
        result["runup"][threshold] = {"angles": angles.tolist(), "measured_m": measured.tolist(), "model_m": model_runup.tolist(),
                                      "meanAbsRelativeError": float(np.mean(np.abs(rel))), "maxAbsRelativeError": float(np.max(np.abs(rel))),
                                      "back90RelativeError": float(rel[back]), "backWithin20Percent": bool(abs(rel[back]) <= 0.2)}
    return result


def monai_bed(dx=0.014):
    """Bed elevation (m, still water at 0) on the 393 x 244 node grid of the BP7 bathymetry file."""
    a = numeric_rows(LAB7 / "Benchmark_2_Bathymetry.txt", 3)
    i, j = np.rint(a[:, 0] / dx).astype(int), np.rint(a[:, 1] / dx).astype(int)
    bed = np.full((j.max() + 1, i.max() + 1), np.nan)
    bed[j, i] = -a[:, 2]
    return bed, dx


def lab7():
    a = numeric_rows(LAB7 / "output_ch5-7-9.csv", 4)
    return a[:, 0], {"5": a[:, 1] / 100.0, "7": a[:, 2] / 100.0, "9": a[:, 3] / 100.0}


def observed_runup7():
    rows = []
    for line in (LAB7 / "OBS_RUNUP.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        if len(values) == 8:
            rows.append(values)
    return np.array(rows)  # x, y, runup of six experimental cases


def bp7(run):
    tl, lab = lab7()
    tm = np.array(run["times"])
    rows = {}
    for g in ["5", "7", "9"]:
        m = np.array(run["series"][g])
        tp_l, ap_l = peak(tl, lab[g], (10, 25))
        tp_m, ap_m = peak(tm, m, (10, 25))
        window = (tl >= 10) & (tl <= 25)
        mi = np.interp(tl[window], tm, m)
        rows[g] = {"labPeak_m": float(ap_l), "modelPeak_m": float(ap_m), "peakError": float((ap_m - ap_l) / ap_l),
                   "labPeakTime_s": float(tp_l), "modelPeakTime_s": float(tp_m), "peakTimeError_s": float(tp_m - tp_l),
                   "nrmsd": float(np.sqrt(np.mean((mi - lab[g][window]) ** 2)) / ap_l), "labInitial_m": float(lab[g][0])}
    box, dx = run["box"], run["dx"]
    maxd = decode(run["maxDepth"], (box["h"], box["w"]))
    bed = decode(run["bed"], (box["h"], box["w"]))
    obs = observed_runup7()
    runup = []
    for threshold in (0.0005, 0.001, 0.002):
        wet = maxd > threshold
        entries = []
        for x, y, *cases in obs:
            # Along the row through y, the highest bed reached by water near the observed x (+-0.1 m).
            j = int(round(y / dx)) - box["j0"]
            i0, i1 = int(round((x - 0.1) / dx)) - box["i0"], int(round((x + 0.1) / dx)) - box["i0"]
            row = slice(max(0, i0), min(box["w"], i1 + 1))
            reach = bed[j, row][wet[j, row]]
            model = float(reach.max()) if reach.size else float("nan")
            entries.append({"x": x, "y": y, "measured_m": cases, "measuredMean_m": float(np.mean(cases)), "model_m": model,
                            "relativeErrorToMean": float((model - np.mean(cases)) / np.mean(cases))})
        valley = (np.arange(box["w"]) * dx + box["i0"] * dx >= 4.95)[None, :] & \
                 ((np.arange(box["h"]) * dx + box["j0"] * dx >= 1.7) & (np.arange(box["h"]) * dx + box["j0"] * dx <= 2.05))[:, None]
        vmax = float(bed[wet & valley].max()) if (wet & valley).any() else float("nan")
        runup.append({"threshold_m": threshold, "points": entries, "valleyMaxRunup_m": vmax,
                      "valleyErrorToObservedMax": float((vmax - obs[0, 2:].mean()) / obs[0, 2:].mean())})
    return {"manning": run["manning"], "retries": run["counters"]["retries"], "gauges": rows, "runup": runup}


if __name__ == "__main__":
    out = {"note": "Computed by tools/nthmp/analysis.py from data/figures/nthmp-*.json and data/nthmp."}
    p6, p7 = FIELDS / "nthmp-bp6.json", FIELDS / "nthmp-bp7.json"
    if p6.exists():
        report = json.loads(p6.read_text(encoding="utf-8"))
        out["bp6"] = {"adapter": report.get("adapter"), "userAgent": report.get("userAgent"), "errors": report.get("errors"),
                      "runs": [bp6(r) for r in report["runs"]]}
    if p7.exists():
        report = json.loads(p7.read_text(encoding="utf-8"))
        out["bp7"] = {"adapter": report.get("adapter"), "errors": report.get("errors"), "runs": [bp7(r) for r in report["runs"]]}
    target = ROOT / "docs" / "results" / "nthmp.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", target)
