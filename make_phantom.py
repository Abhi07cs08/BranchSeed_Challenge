#!/usr/bin/env python3
"""
make_phantom.py — build a full-size synthetic Branchseed case WITH ground truth.

Why this exists: you cannot calibrate a detector without labels, and the dev subset is
small. This writes a 512x512x174 case at realistic spacing containing 15 known daughters
plus every distractor that matters (IVC touching the aorta, calcified plaque on the wall,
contrast-filled kidneys at the end of the renal arteries, vertebral bone, bowel gas),
with partial-volume ramps and a reconstruction blur so thin vessels are genuinely dim.

    python make_phantom.py --out phantom
    -> phantom/data/phantom01/orig1.nii.gz
       phantom/data/phantom01/mask1.nii.gz
       phantom/refs/phantom01.json

Then:
    python run.py --image phantom/data/phantom01/orig1.nii.gz \
                  --aorta-mask phantom/data/phantom01/mask1.nii.gz \
                  --output preds/phantom01.json --viz viz/phantom01.png
    python evaluate.py --pred preds/ --ref phantom/refs/ --match-mm 3,5,8 --per-case
"""

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# ----------------------------------------------------------------- grid definition
SIZE_XYZ = (512, 512, 174)
SPACING_XYZ = (0.78, 0.78, 1.50)
ORIGIN_XYZ = (-199.0, -199.0, -130.0)      # arbitrary but non-trivial, like real data

HU = dict(air=-1000.0, pad=-2048.0, fat=-90.0, soft=45.0, bone=620.0,
          lumen=320.0, vein=135.0, calcium=950.0, kidney=250.0, gas=-900.0)

AORTA_R = 10.0          # mm
AORTA_XY = (0.0, 30.0)  # aorta axis position in physical x, y (y+ = posterior here)

# daughters: (name, z_mm, direction (x,y,z), radius_mm, length_mm)
#   direction is from the ostium outward; -y is anterior
DAUGHTERS = [
    ("celiac",       95.0, ( 0.00, -1.00,  0.05), 3.5, 16.0),   # common trunk, splits below
    ("sma",          75.0, ( 0.05, -1.00, -0.35), 3.2, 45.0),
    ("renal_l",      45.0, (-1.00,  0.05,  0.00), 2.5, 20.0),
    ("renal_r",      42.0, ( 1.00,  0.05,  0.00), 2.5, 20.0),
    ("acc_renal_r",  30.0, ( 1.00,  0.10, -0.30), 1.5, 24.0),
    ("gonadal_l",     8.0, (-0.45, -0.30, -1.00), 1.0, 30.0),
    ("ima",         -40.0, (-0.30, -1.00, -0.20), 1.3, 30.0),
]
# posterior lumbar pairs — small, numerous, and where recall is won or lost
for _z in (70.0, 40.0, 10.0, -20.0):
    DAUGHTERS.append((f"lumbar_l_{int(_z)}", _z, (-0.60, 0.80, 0.0), 0.85, 20.0))
    DAUGHTERS.append((f"lumbar_r_{int(_z)}", _z, ( 0.60, 0.80, 0.0), 0.85, 20.0))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Phantom:
    """Composites structures with soft (partial-volume) edges in physical mm."""

    def __init__(self, size_xyz, spacing_xyz, origin_xyz):
        self.nx, self.ny, self.nz = size_xyz
        self.sp = np.asarray(spacing_xyz, float)          # x, y, z
        self.org = np.asarray(origin_xyz, float)
        self.vol = np.full((self.nz, self.ny, self.nx), HU["air"], np.float32)
        self.aorta = np.zeros((self.nz, self.ny, self.nx), bool)
        self.vox = float(self.sp.mean())

    # physical mm of a voxel centre, per axis
    def ax(self, which):
        n = dict(x=self.nx, y=self.ny, z=self.nz)[which]
        i = "xyz".index(which)
        return self.org[i] + np.arange(n) * self.sp[i]

    def bbox(self, lo_mm, hi_mm, pad_mm=3.0):
        """Index slices covering a physical box, clipped to the grid."""
        out = []
        for i, which in enumerate("zyx"):
            j = "xyz".index(which)
            a = int(np.floor((lo_mm[j] - pad_mm - self.org[j]) / self.sp[j]))
            b = int(np.ceil((hi_mm[j] + pad_mm - self.org[j]) / self.sp[j])) + 1
            n = dict(x=self.nx, y=self.ny, z=self.nz)[which]
            out.append(slice(max(a, 0), min(b, n)))
        return tuple(out)

    def _coords(self, sl):
        z = self.ax("z")[sl[0]][:, None, None]
        y = self.ax("y")[sl[1]][None, :, None]
        x = self.ax("x")[sl[2]][None, None, :]
        return x, y, z

    def paint(self, sl, occ, value, also_aorta=False):
        occ = np.clip(occ, 0.0, 1.0).astype(np.float32)
        self.vol[sl] = self.vol[sl] * (1 - occ) + value * occ
        if also_aorta:
            self.aorta[sl] |= occ > 0.5

    def capsule(self, a_mm, b_mm, r, value, also_aorta=False):
        """Cylinder with hemispherical caps from a to b, soft-edged."""
        a, b = np.asarray(a_mm, float), np.asarray(b_mm, float)
        lo = np.minimum(a, b) - r
        hi = np.maximum(a, b) + r
        sl = self.bbox(lo, hi)
        x, y, z = self._coords(sl)
        ab = b - a
        L2 = float(ab @ ab)
        px, py, pz = x - a[0], y - a[1], z - a[2]
        t = (px * ab[0] + py * ab[1] + pz * ab[2]) / max(L2, 1e-9)
        t = np.clip(t, 0.0, 1.0)
        dx = px - t * ab[0]
        dy = py - t * ab[1]
        dz = pz - t * ab[2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        self.paint(sl, 0.5 + (r - d) / self.vox, value, also_aorta)

    def ball(self, c_mm, r, value):
        c = np.asarray(c_mm, float)
        sl = self.bbox(c - r, c + r)
        x, y, z = self._coords(sl)
        d = np.sqrt((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2)
        self.paint(sl, 0.5 + (r - d) / self.vox, value)

    def ellipsoid_body(self, c_mm, radii, value):
        c, rr = np.asarray(c_mm, float), np.asarray(radii, float)
        sl = self.bbox(c - rr, c + rr)
        x, y, z = self._coords(sl)
        q = ((x - c[0]) / rr[0])**2 + ((y - c[1]) / rr[1])**2 + ((z - c[2]) / rr[2])**2
        self.paint(sl, (1.05 - q) / 0.05, value)

    def to_sitk(self):
        img = sitk.GetImageFromArray(self.vol)
        img.SetSpacing(SPACING_XYZ); img.SetOrigin(ORIGIN_XYZ)
        msk = sitk.GetImageFromArray(self.aorta.astype(np.uint8))
        msk.SetSpacing(SPACING_XYZ); msk.SetOrigin(ORIGIN_XYZ)
        return img, msk


def build(seed=0, noise_hu=18.0, blur_vox=0.6):
    rng = np.random.default_rng(seed)
    p = Phantom(SIZE_XYZ, SPACING_XYZ, ORIGIN_XYZ)
    ax, ay = AORTA_XY
    z0, z1 = p.ax("z")[0], p.ax("z")[-1]

    # body: fat rind then soft tissue
    p.ellipsoid_body((0, 20, 0), (165, 120, 200), HU["fat"])
    p.ellipsoid_body((0, 20, 0), (152, 108, 200), HU["soft"])
    # vertebral column, posterior
    p.capsule((ax, ay + 42, z0), (ax, ay + 42, z1), 19.0, HU["bone"])
    # bowel gas
    for _ in range(7):
        p.ball((rng.uniform(-90, 90), rng.uniform(-70, 0), rng.uniform(z0 + 20, z1 - 20)),
               rng.uniform(7, 16), HU["gas"])
    # IVC: parallel to the aorta and just touching it
    p.capsule((ax + AORTA_R + 11.2, ay, z0), (ax + AORTA_R + 11.2, ay, z1), 11.0, HU["vein"])

    # aorta, running the full z extent -> BOTH cropped end caps present
    p.capsule((ax, ay, z0 - 5), (ax, ay, z1 + 5), AORTA_R, HU["lumen"], also_aorta=True)

    # daughters
    ref = []
    for name, zc, d, r, L in DAUGHTERS:
        u = unit(d)
        axis = np.array([ax, ay, zc])
        # ostium: where the branch centreline crosses the aortic surface
        radial = unit([u[0], u[1], 0.0]) if abs(u[0]) + abs(u[1]) > 1e-6 else np.array([1.0, 0, 0])
        ost = axis + radial * AORTA_R
        start = axis + radial * (AORTA_R - 2.0)        # overlap so lumens are continuous
        end = ost + u * L
        p.capsule(start, end, r, HU["lumen"])
        ref.append(dict(name=name, ostium=ost, seed=ost + u * 5.0, radius=r, direction=u))

    # celiac trifurcates 16 mm out: ONE aortic ostium, three vessels after it
    cel = next(x for x in ref if x["name"] == "celiac")
    hub = cel["ostium"] + cel["direction"] * 16.0
    for sub in [(-0.8, -0.5, 0.4), (0.1, -0.9, -0.3), (0.8, -0.4, 0.2)]:
        p.capsule(hub, hub + unit(sub) * 28.0, 2.0, HU["lumen"])

    # contrast-filled kidneys at the end of each renal artery: the leak trap
    for nm in ("renal_l", "renal_r"):
        k = next(x for x in ref if x["name"] == nm)
        p.ball(k["ostium"] + k["direction"] * 44.0, 24.0, HU["kidney"])

    # calcified plaque sitting on the aortic wall
    p.ball((ax - AORTA_R * 0.8, ay - AORTA_R * 0.6, -10.0), 3.2, HU["calcium"])

    # noise, reconstruction blur, FOV padding
    p.vol += rng.normal(0.0, noise_hu, p.vol.shape).astype(np.float32)
    p.vol = ndimage.gaussian_filter(p.vol, blur_vox).astype(np.float32)
    xs, ys = np.meshgrid(p.ax("x"), p.ax("y"), indexing="xy")
    outside = (xs**2 + (ys - 20)**2) > 195.0**2
    p.vol[:, outside] = HU["pad"]
    np.clip(p.vol, HU["pad"], 3071.0, out=p.vol)
    return p, ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="phantom")
    ap.add_argument("--case-id", default="phantom01")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    p, ref = build(a.seed)
    dd = os.path.join(a.out, "data", a.case_id)
    rd = os.path.join(a.out, "refs")
    os.makedirs(dd, exist_ok=True); os.makedirs(rd, exist_ok=True)

    img, msk = p.to_sitk()
    sitk.WriteImage(img, os.path.join(dd, "orig1.nii.gz"), True)
    sitk.WriteImage(msk, os.path.join(dd, "mask1.nii.gz"), True)

    payload = {"case_id": a.case_id, "parent": {"instance_id": "aorta"}, "daughters": [
        {"instance_id": f"branch_{i:03d}", "parent_instance_id": "aorta",
         "truth_name": r["name"],
         "ostium_xyz_mm": [round(float(v), 3) for v in r["ostium"]],
         "seed_xyz_mm": [round(float(v), 3) for v in r["seed"]],
         "radius_mm": round(float(r["radius"]), 3),
         "direction_xyz": [round(float(v), 5) for v in r["direction"]]}
        for i, r in enumerate(ref, 1)]}
    with open(os.path.join(rd, f"{a.case_id}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # A second reference set holding only the branches the published rule makes eligible
    # (origin diameter >= 2 mm). The phantom deliberately contains sub-2 mm lumbars so the
    # detector can be exercised on them, but scoring against those would report a recall
    # failure for behaviour the brief actually requires.
    rd2 = os.path.join(a.out, "refs_eligible")
    os.makedirs(rd2, exist_ok=True)
    elig = dict(payload)
    elig["daughters"] = [d for d in payload["daughters"] if d["radius_mm"] * 2 >= 2.0]
    with open(os.path.join(rd2, f"{a.case_id}.json"), "w") as f:
        json.dump(elig, f, indent=2)

    print(f"{a.case_id}: {SIZE_XYZ} @ {SPACING_XYZ} mm, "
          f"aorta {p.aorta.sum()} voxels, {len(ref)} ground-truth daughters")
    print(f"  {dd}/orig1.nii.gz + mask1.nii.gz")
    print(f"  {rd}/{a.case_id}.json  (all {len(ref)})")
    print(f"  {rd2}/{a.case_id}.json  ({len(elig['daughters'])} eligible at >=2mm)")


if __name__ == "__main__":
    main()
