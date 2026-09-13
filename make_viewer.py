#!/usr/bin/env python3
"""
make_viewer.py — build the data bundle the clinician viewer reads.

For every case it runs the SAME run.py the submission uses (no viewer-only tuning), and
writes three small files per case:

    ct.json        the cropped CT ROI (gzipped NIfTI, base64) in world coordinates
    seg.json       label volume: 1 = parent aorta, 2.. = each traced daughter
    pred.json      the submitted prediction, verbatim

The two volumes are carried inside JSON rather than served as .nii.gz because the viewer
has to run from a plain static folder -- and from a sandbox that serves only standard web
media types -- with no server configuration at all. NiiVue reads them with
NVImage.loadFromBase64, so it is the same NIfTI either way.

plus a cases.json index the page loads first. The ROI crop is what keeps this honest AND
small: a whole abdominal study is 30-80 MB, the ROI around the aorta is under a megabyte
gzipped, and it carries its own origin so it still lands in the right place in world space.

  python make_viewer.py --data EVAL_SET --refs evalrefs --out viewer
  python make_viewer.py --data data --out viewer --append
"""
import argparse, base64, glob, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def find_pair(case_dir):
    """Accept any sane naming: orig*/img*/ct* for the image, mask*/aorta*/seg* for the mask."""
    files = sorted(glob.glob(os.path.join(case_dir, "*.nii*")))
    if not files:
        return None, None
    img = msk = None
    for f in files:
        b = os.path.basename(f).lower()
        if any(k in b for k in ("mask", "aorta", "seg", "label")) and "daughter" not in b:
            msk = msk or f
        elif any(k in b for k in ("orig", "img", "image", "ct", "vol")):
            img = img or f
    if img is None or msk is None:
        cand = [f for f in files if "daughter" not in os.path.basename(f).lower()]
        if len(cand) == 2:
            a, b = cand
            img, msk = (a, b) if "mask" in os.path.basename(b).lower() else (b, a)
    return img, msk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory of case subdirectories")
    ap.add_argument("--refs", default=None, help="reference JSONs, to show expert agreement")
    ap.add_argument("--out", default=os.path.join(HERE, "viewer"))
    ap.add_argument("--match-mm", type=float, default=5.0)
    ap.add_argument("--append", action="store_true", help="add to an existing bundle")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data_root = os.path.join(args.out, "data")
    os.makedirs(data_root, exist_ok=True)
    index_path = os.path.join(args.out, "cases.json")
    cases = []
    if args.append and os.path.exists(index_path):
        cases = json.load(open(index_path))["cases"]
    have = {c["id"] for c in cases}

    dirs = sorted(d for d in glob.glob(os.path.join(args.data, "*")) if os.path.isdir(d))
    if args.limit:
        dirs = dirs[:args.limit]

    for case_dir in dirs:
        cid = os.path.basename(case_dir)
        if cid in have:
            continue
        img, msk = find_pair(case_dir)
        if not img or not msk:
            print(f"  {cid}: skipped (could not identify image/mask)", file=sys.stderr)
            continue
        out_dir = os.path.join(data_root, cid)
        os.makedirs(out_dir, exist_ok=True)
        pred_p = os.path.join(out_dir, "pred.json")
        ct_nii = os.path.join(out_dir, "_ct.nii.gz")
        seg_nii = os.path.join(out_dir, "_seg.nii.gz")
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"),
                            "--image", img, "--aorta-mask", msk,
                            "--output", pred_p,
                            "--overlay", seg_nii, "--roi-image", ct_nii],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {cid}: FAILED\n{r.stderr[-800:]}", file=sys.stderr)
            shutil.rmtree(out_dir, ignore_errors=True)
            continue
        secs = time.time() - t0
        for src, dst in ((ct_nii, "ct.json"), (seg_nii, "seg.json")):
            with open(src, "rb") as fh:
                blob = base64.b64encode(fh.read()).decode("ascii")
            json.dump({"name": os.path.basename(src).lstrip("_"), "b64": blob},
                      open(os.path.join(out_dir, dst), "w"))
            os.remove(src)
        pred = json.load(open(pred_p))
        pred["case_id"] = cid
        json.dump(pred, open(pred_p, "w"), indent=2)

        entry = {
            "id": cid,
            "n": len(pred["daughters"]),
            "seconds": round(secs, 2),
            "ct": f"data/{cid}/ct.json",
            "seg": f"data/{cid}/seg.json",
            "pred": f"data/{cid}/pred.json",
            "source_image": os.path.basename(img),
            "source_mask": os.path.basename(msk),
        }
        m = re.search(r"grid \((\d+), (\d+), (\d+)\)", r.stdout or "")
        entry["daughters"] = [{
            "id": d["instance_id"],
            "label": i + 2,                      # matches the value written into seg.nii.gz
            "ostium": d["ostium_xyz_mm"],
            "seed": d["seed_xyz_mm"],
            "radius": d["radius_mm"],
            "direction": d["direction_xyz"],
        } for i, d in enumerate(pred["daughters"])]

        ref_p = os.path.join(args.refs, f"{cid}.json") if args.refs else None
        if ref_p and os.path.exists(ref_p):
            entry["reference"] = score_against(entry["daughters"], ref_p, args.match_mm)
        cases.append(entry)
        sz = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
        print(f"  {cid}: {entry['n']} daughters, {secs:.1f}s, {sz/1e6:.2f} MB")

    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "cases": cases},
              open(index_path, "w"), indent=2)
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(args.out) for f in fs)
    print(f"\nwrote {index_path}  ({len(cases)} cases, {total/1e6:.1f} MB total)")


def score_against(daughters, ref_path, match_mm):
    """One-to-one match against the draft expert annotation, so the viewer can show a
    clinician which detections the reference agrees with -- and, just as usefully, which
    detections sit outside the part of the aorta anybody annotated."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ref = json.load(open(ref_path))["daughters"]
    R = np.array([d["ostium_xyz_mm"] for d in ref], float)
    P = np.array([d["ostium"] for d in daughters], float)
    out = {"n_ref": len(R), "matched": 0, "match_mm": match_mm, "errors": {}}
    if not len(R) or not len(P):
        return out
    C = np.linalg.norm(R[:, None, :] - P[None, :, :], axis=2)
    ri, pi = linear_sum_assignment(C)
    for a, b in zip(ri, pi):
        if C[a, b] <= match_mm:
            out["errors"][daughters[b]["id"]] = round(float(C[a, b]), 2)
            out["matched"] += 1
    return out


if __name__ == "__main__":
    sys.exit(main())
