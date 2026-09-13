#!/usr/bin/env python3
"""
audit.py — look at EVERY candidate, accepted and rejected, one at a time.

Without reference annotations you cannot measure anything, and a whole-aorta MIP is far
too coarse to judge a 2 mm vessel. This renders each candidate as its own zoomed triptych
around its ostium, numbered, with the measured features and the accept/reject reason
printed underneath. You mark which ones are genuine; the script then writes reference
JSONs from your marks so `evaluate.py` can score against them.

Rejected candidates are shown too — that is where your false negatives are hiding.

    # 1. render the sheets and a blank marking file
    python audit.py --image data/subject001/orig1.nii --aorta-mask data/subject001/mask1.nii \
                    --out audit/subject001

    # 2. open audit/subject001/sheet_*.png, then edit audit/subject001/marks.csv
    #    putting 1 in is_real for every candidate that is a genuine aortic branch

    # 3. turn your marks into a reference file
    python audit.py --make-refs audit/subject001/marks.csv --refs-dir refs/

Then `python evaluate.py --pred preds/ --ref refs/ --match-mm 3,5` measures you for real.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import run as bs


def build_all_candidates(case, args):
    """Mirror run.main()'s flow but keep every candidate, with its verdict."""
    labels, lab = bs.find_candidates(case, args.collar_mm, args.rind_mm, args.touch_mm,
                                     args.min_cand_voxels, False)
    items, alive = [], []
    for li in labels:
        c, why = bs.grow(case, lab, li, args)
        if c is None:
            items.append(dict(region=(lab == li), verdict=why, accepted=0, c=None))
            continue
        grown = c["grown"]
        c, why = bs.place_ostium(case, c, args)
        if c is None:
            items.append(dict(region=grown, verdict=why, accepted=0, c=None))
            continue
        alive.append(c)
    alive = bs.merge_trunks(case, alive, args.merge_mm)
    for c in alive:
        c2, why = bs.measure(case, c, args)
        if c2 is None:
            items.append(dict(region=c["grown"], verdict=why, accepted=0, c=c))
        else:
            items.append(dict(region=c2["grown"], verdict="ACCEPTED", accepted=1, c=c2))
    # sort superior -> inferior where we know the position
    def key(it):
        c = it["c"]
        return -(c["ostium_mm"][2] if c is not None and "ostium_mm" in c else -1e9)
    items.sort(key=key)
    return items


def tile(ax, vol, mask, region, centre, spacing, axis, half_mm, lumen, soft):
    """One zoomed orthogonal slab around `centre`, with mask outline and candidate fill."""
    a0, a1 = [k for k in (0, 1, 2) if k != axis]
    half = np.maximum(np.round(half_mm / spacing).astype(int), 4)
    thick = max(int(round(4.0 / spacing[axis])), 1)
    c = np.round(centre).astype(int)
    sl = [slice(max(c[k] - half[k], 0), min(c[k] + half[k] + 1, vol.shape[k])) for k in range(3)]
    sl[axis] = slice(max(c[axis] - thick, 0), min(c[axis] + thick + 1, vol.shape[axis]))
    sl = tuple(sl)
    img = vol[sl].max(axis=axis)
    ax.imshow(img, cmap="gray", vmin=soft - 100, vmax=lumen * 1.05,
              aspect=spacing[a0] / spacing[a1])
    mk = mask[sl].max(axis=axis).astype(float)
    if mk.any():
        ax.contour(mk, levels=[0.5], colors="#3fa7c4", linewidths=1.0)
    if region is not None:
        rg = region[sl].max(axis=axis).astype(float)
        ax.imshow(np.ma.masked_where(rg < 0.5, rg), cmap="autumn", alpha=0.5,
                  aspect=spacing[a0] / spacing[a1], vmin=0, vmax=1)
    ax.plot([c[a1] - sl[a1].start], [c[a0] - sl[a0].start], "o", ms=9, mfc="none",
            mec="#e8443f", mew=1.8)
    ax.set_xticks([]); ax.set_yticks([])


def render(case, items, out_dir, per_sheet=6, half_mm=22.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    vol, mask, sp = case["vol"], case["mask"], case["spacing"]
    names = ["axial", "coronal", "sagittal"]
    sheets = []
    for start in range(0, len(items), per_sheet):
        chunk = items[start:start + per_sheet]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(9.0, 3.05 * len(chunk)), squeeze=False)
        for row, it in enumerate(chunk):
            n = start + row + 1
            c = it["c"]
            if c is not None and "ostium_idx" in c:
                centre = np.asarray(c["ostium_idx"], float)
            elif it["region"] is not None and it["region"].any():
                centre = np.argwhere(it["region"]).mean(axis=0)
            else:
                for a in axes[row]:
                    a.axis("off")
                continue
            for col in range(3):
                tile(axes[row][col], vol, mask, it["region"], centre, sp, col, half_mm,
                     case["lumen_hu"], case["soft_hu"])
                if row == 0:
                    axes[row][col].set_title(names[col], fontsize=9)
            bits = [f"#{n}"]
            if c is not None:
                for k, lab in (("d_min", "cal"), ("anisotropy", "anis"), ("leak", "leak"),
                               ("bright_ratio", "bright"), ("ostium_diam_mm", "ost"),
                               ("radius_mm", "r")):
                    if k in c:
                        bits.append(f"{lab} {c[k]:.2f}")
            tag = "ACCEPTED" if it["accepted"] else f"rejected: {it['verdict']}"
            axes[row][0].set_ylabel(f"#{n}\n{'OK' if it['accepted'] else 'rej'}",
                                    fontsize=11, rotation=0, labelpad=26,
                                    color="#1a7f37" if it["accepted"] else "#b3202e")
            axes[row][1].set_xlabel("   ".join(bits[1:])[:96], fontsize=7.5)
            axes[row][2].set_xlabel(tag[:64], fontsize=7.5,
                                    color="#1a7f37" if it["accepted"] else "#b3202e")
        fig.tight_layout()
        p = os.path.join(out_dir, f"sheet_{start // per_sheet + 1:02d}.png")
        fig.savefig(p, dpi=115)
        plt.close(fig)
        sheets.append(p)
    return sheets


def write_marks(case_id, items, out_dir):
    p = os.path.join(out_dir, "marks.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "is_real", "case_id", "auto_accepted", "verdict",
                    "ostium_x", "ostium_y", "ostium_z", "seed_x", "seed_y", "seed_z",
                    "radius_mm", "dir_x", "dir_y", "dir_z"])
        for i, it in enumerate(items, 1):
            c = it["c"]
            row = [i, "", case_id, it["accepted"], it["verdict"][:60]]
            if c is not None and "ostium_mm" in c:
                row += [round(float(v), 3) for v in c["ostium_mm"]]
                row += [round(float(v), 3) for v in c["seed_mm"]]
                row += [round(float(c["radius_mm"]), 3)]
                row += [round(float(v), 5) for v in c["direction"]]
            else:
                row += [""] * 10
            w.writerow(row)
    return p


def make_refs(marks_csv, refs_dir):
    rows = list(csv.DictReader(open(marks_csv)))
    by = {}
    for r in rows:
        if str(r["is_real"]).strip() not in ("1", "y", "Y", "yes", "true"):
            continue
        if not r["ostium_x"]:
            print(f"  skipping #{r['n']}: marked real but has no measured ostium — "
                  f"it was rejected before measurement, so there is nothing to score against",
                  file=sys.stderr)
            continue
        by.setdefault(r["case_id"], []).append(r)
    os.makedirs(refs_dir, exist_ok=True)
    for case_id, rs in by.items():
        payload = {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": [
            {"instance_id": f"branch_{i:03d}", "parent_instance_id": "aorta",
             "ostium_xyz_mm": [float(r["ostium_x"]), float(r["ostium_y"]), float(r["ostium_z"])],
             "seed_xyz_mm": [float(r["seed_x"]), float(r["seed_y"]), float(r["seed_z"])],
             "radius_mm": float(r["radius_mm"]),
             "direction_xyz": [float(r["dir_x"]), float(r["dir_y"]), float(r["dir_z"])]}
            for i, r in enumerate(rs, 1)]}
        p = os.path.join(refs_dir, f"{case_id}.json")
        json.dump(payload, open(p, "w"), indent=2)
        print(f"wrote {p}  ({len(rs)} branches marked real)")
    if not by:
        print("nothing marked real — put 1 in the is_real column first", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-refs", help="a filled-in marks.csv to convert")
    ap.add_argument("--refs-dir", default="refs")
    ap.add_argument("--image")
    ap.add_argument("--aorta-mask", dest="mask")
    ap.add_argument("--out", default=None)
    ap.add_argument("--half-mm", type=float, default=22.0, help="zoom half-width per tile")
    ap.add_argument("--per-sheet", type=int, default=6)
    known, rest = ap.parse_known_args()

    if known.make_refs:
        make_refs(known.make_refs, known.refs_dir)
        return 0
    if not (known.image and known.mask):
        ap.error("give --image and --aorta-mask (or --make-refs)")

    args = bs.parse_args(["--image", known.image, "--aorta-mask", known.mask,
                          "--output", os.devnull] + rest)
    case = bs.load_and_crop(args.image, args.mask, args.margin_mm, False)
    case = bs.intensity_model(case, args.core_erode_mm, args.thr_frac, args.ceiling_frac,
                              args.hu_ceiling, args.hu_ceiling_max, args.lumen_pct, False)
    case = bs.aorta_geometry(case, args.cap_margin_mm, args.cap_cos, False)
    items = build_all_candidates(case, args)

    case_id = os.path.basename(os.path.dirname(os.path.abspath(known.image))) or "case"
    out = known.out or os.path.join("audit", case_id)
    sheets = render(case, items, out, known.per_sheet, known.half_mm)
    marks = write_marks(case_id, items, out)
    n_acc = sum(it["accepted"] for it in items)
    print(f"{case_id}: {len(items)} candidates ({n_acc} auto-accepted, {len(items)-n_acc} rejected)")
    for p in sheets:
        print(f"  {p}")
    print(f"  {marks}   <- put 1 in is_real for every genuine branch, then:")
    print(f"  python audit.py --make-refs {marks} --refs-dir refs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
