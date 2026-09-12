#!/usr/bin/env python3
"""
evaluate.py — local scorer for Branchseed, using the organisers' stated semantics.

    python evaluate.py --pred preds/ --ref data_dev_refs/ --match-mm 5

The brief says: "Predictions will be matched to references one-to-one, so duplicate
detections count as false positives." That is a bipartite assignment problem, not a
greedy nearest-neighbour loop — a greedy matcher will flatter you. This uses
scipy.optimize.linear_sum_assignment on ostium distance, then keeps only assignments
inside --match-mm.

Reports the four things you can actually control, then a composite that mirrors the
published weights (45 / 25 / 15) so one number moves when you tune a parameter.

The match radius is NOT published. Sweep it (--match-mm 3,5,8,10) and make sure your
ranking of configurations is stable across the sweep; if it is not, you are tuning noise.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    sys.exit("need scipy: pip install scipy")


def load_daughters(path):
    with open(path) as f:
        d = json.load(f)
    case = d.get("case_id") or os.path.splitext(os.path.basename(path))[0]
    out = []
    for x in d.get("daughters", []):
        rec = {"id": x.get("instance_id")}
        if "ostium_xyz_mm" in x:
            rec["ostium"] = np.asarray(x["ostium_xyz_mm"], float)
        elif "centreline" in x and len(x["centreline"]):
            rec["ostium"] = np.asarray(x["centreline"][0], float)
        else:
            continue
        for k, dst in (("seed_xyz_mm", "seed"), ("direction_xyz", "dir")):
            if k in x:
                rec[dst] = np.asarray(x[k], float)
        if "radius_mm" in x:
            rec["radius"] = float(x["radius_mm"])
        if "centreline" in x:
            rec["centreline"] = np.asarray(x["centreline"], float)
        out.append(rec)
    return case, out


def index_dir(spec):
    """Accept a directory, a glob, or a single file. Returns {case_id: [daughters]}."""
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "**", "*.json"), recursive=True))
    else:
        files = sorted(glob.glob(spec))
    idx = {}
    for f in files:
        try:
            case, ds = load_daughters(f)
        except Exception as e:
            print(f"[skip] {f}: {e}", file=sys.stderr)
            continue
        idx[case] = ds
    return idx


def match(pred, ref, match_mm):
    """One-to-one assignment on ostium distance. Returns (pairs, n_fp, n_fn, dists)."""
    if not pred or not ref:
        return [], len(pred), len(ref), []
    P = np.stack([p["ostium"] for p in pred])
    R = np.stack([r["ostium"] for r in ref])
    D = np.linalg.norm(P[:, None, :] - R[None, :, :], axis=2)
    big = match_mm * 1000.0
    C = np.where(D <= match_mm, D, big)
    ri, ci = linear_sum_assignment(C)
    pairs, dists = [], []
    for i, j in zip(ri, ci):
        if D[i, j] <= match_mm:
            pairs.append((i, j))
            dists.append(float(D[i, j]))
    return pairs, len(pred) - len(pairs), len(ref) - len(pairs), dists


def seed_on_branch(p, r, tol_mm):
    """Is the predicted seed near the reference proximal centreline?"""
    if "seed" not in p:
        return None
    if "centreline" in r and len(r["centreline"]) > 1:
        d = np.linalg.norm(r["centreline"] - p["seed"][None, :], axis=1).min()
    elif "seed" in r:
        d = float(np.linalg.norm(r["seed"] - p["seed"]))
    else:
        return None
    return float(d), bool(d <= tol_mm)


def angle_deg(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="dir, glob or file of prediction JSONs")
    ap.add_argument("--ref", required=True, help="dir, glob or file of reference JSONs")
    ap.add_argument("--match-mm", default="5", help="match radius in mm; comma-separate to sweep")
    ap.add_argument("--seed-tol-mm", type=float, default=3.0)
    ap.add_argument("--per-case", action="store_true")
    a = ap.parse_args()

    preds, refs = index_dir(a.pred), index_dir(a.ref)
    cases = sorted(set(refs) & set(preds))
    missing = sorted(set(refs) - set(preds))
    extra = sorted(set(preds) - set(refs))
    if missing:
        print(f"WARNING: no prediction for {len(missing)} reference case(s): {missing[:5]}", file=sys.stderr)
    if extra:
        print(f"note: {len(extra)} prediction(s) with no reference, ignored", file=sys.stderr)
    if not cases:
        sys.exit("no overlapping case_ids between --pred and --ref")

    for mm in [float(x) for x in str(a.match_mm).split(",")]:
        TP = FP = FN = 0
        dists, angles, rad_err, seed_d, seed_ok = [], [], [], [], []
        rows = []
        for c in cases:
            P, R = preds[c], refs[c]
            pairs, fp, fn, dd = match(P, R, mm)
            TP += len(pairs); FP += fp; FN += fn
            dists += dd
            for i, j in pairs:
                p, r = P[i], R[j]
                if "dir" in p and "dir" in r:
                    angles.append(angle_deg(p["dir"], r["dir"]))
                if "radius" in p and "radius" in r:
                    rad_err.append(abs(p["radius"] - r["radius"]))
                s = seed_on_branch(p, r, a.seed_tol_mm)
                if s:
                    seed_d.append(s[0]); seed_ok.append(s[1])
            rows.append((c, len(R), len(P), len(pairs), fp, fn,
                         np.mean(dd) if dd else float("nan")))
        for c in missing:
            FN += len(refs[c])

        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ost = float(np.mean([1 - min(d / mm, 1.0) for d in dists])) if dists else 0.0
        q = []
        if seed_ok: q.append(float(np.mean(seed_ok)))
        if angles:  q.append(float(np.mean([max(0.0, 1 - x / 60.0) for x in angles])))
        if rad_err: q.append(float(np.mean([max(0.0, 1 - e / 1.5) for e in rad_err])))
        qual = float(np.mean(q)) if q else 0.0
        composite = 0.45 * f1 + 0.25 * ost + 0.15 * qual

        print(f"\n{'='*78}\nmatch radius {mm:g} mm   ·   {len(cases)} case(s) scored"
              f"{f' (+{len(missing)} missing, counted as all-FN)' if missing else ''}\n{'='*78}")
        if a.per_case:
            print(f"{'case':<14}{'ref':>5}{'pred':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'mean d':>9}")
            print("-" * 49)
            for c, nr, np_, tp, fp, fn, md in rows:
                print(f"{c:<14}{nr:>5}{np_:>6}{tp:>5}{fp:>5}{fn:>5}{md:>9.2f}")
            print("-" * 49)
        print(f"  discovery   TP {TP}  FP {FP}  FN {FN}")
        print(f"              precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}      [45%]")
        if dists:
            print(f"  ostium      mean {np.mean(dists):.2f} mm   median {np.median(dists):.2f} mm"
                  f"   p90 {np.percentile(dists,90):.2f} mm   score {ost:.3f}   [25%]")
        if seed_ok:
            print(f"  seed        {100*np.mean(seed_ok):.0f}% within {a.seed_tol_mm} mm of the ref path"
                  f"   (mean {np.mean(seed_d):.2f} mm)")
        if angles:
            print(f"  direction   mean {np.mean(angles):.1f} deg   median {np.median(angles):.1f} deg")
        if rad_err:
            print(f"  radius      mean abs err {np.mean(rad_err):.2f} mm")
        print(f"  quality                                              score {qual:.3f}   [15%]")
        print(f"\n  COMPOSITE (45/25/15 of the published rubric)  {composite:.4f}")
        if FP > FN:
            print("  -> over-detecting. Raise --thr-frac or --min-ostium-mm in run.py.")
        elif FN > FP:
            print("  -> under-detecting. Lower --thr-frac, or --min-reach-mm / --min-ostium-mm.")


if __name__ == "__main__":
    main()
