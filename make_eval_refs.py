#!/usr/bin/env python3
"""
make_eval_refs.py — turn the organisers' annotations.json files into scoring references.

The EVAL_SET ships real reference annotations. They are already in the required schema, so
this just lifts the scored fields out into one file per case that evaluate.py can read.

    python make_eval_refs.py --eval ~/Downloads/EVAL_SET --out evalrefs
    python evaluate.py --pred evalpreds --ref evalrefs --match-mm 3,5 --per-case

Note `radius_mm` is often null in the annotations (recorded as "search_envelope_limited").
Where it is, half the manual origin-diameter estimate is substituted so the radius term is
still scorable; that substitution is an approximation, not the organisers' measurement.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True, help="the EVAL_SET directory")
ap.add_argument("--out", default="evalrefs")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
total = 0
for path in sorted(glob.glob(os.path.join(os.path.expanduser(a.eval), "case_*", "annotations.json"))):
    d = json.load(open(path))
    cid = d["case_id"]
    out = {"case_id": cid, "parent": {"instance_id": "aorta"}, "daughters": []}
    for x in d["daughters"]:
        r = x.get("radius_mm")
        if r is None:
            est = x.get("origin_diameter_estimate_mm")
            r = (est / 2.0) if est else None
        out["daughters"].append({
            "instance_id": x["instance_id"],
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": x["ostium_xyz_mm"],
            "seed_xyz_mm": x["seed_xyz_mm"],
            "radius_mm": r,
            "direction_xyz": x["direction_xyz"],
            "centreline": x.get("centerline_xyz_mm", []),
        })
    json.dump(out, open(os.path.join(a.out, f"{cid}.json"), "w"), indent=2)
    total += len(out["daughters"])
    print(f"  {cid}: {len(out['daughters'])} daughters  "
          f"(spacing {d['spacing_xyz_mm']}, policy min origin {d['policy']['minimum_origin_diameter_mm']} mm)")
print(f"wrote {a.out}/  —  {total} reference daughters")
