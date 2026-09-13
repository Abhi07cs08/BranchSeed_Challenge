#!/usr/bin/env python3
"""
V6 BranchSeed detector.

Goal: find every eligible artery that DIRECTLY leaves the supplied abdominal-aorta mask.
The implementation is intentionally geometry-first: candidates must begin at a specific
3-D aortic wall opening, continue as contrast-filled lumen for >=5 mm from that wall,
and show real outward progress instead of merely hugging the wall.

Usage:
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

try:
    from skimage.graph import MCP_Geometric
except ImportError:
    sys.exit("need scikit-image: pip install scikit-image")

HU_CLIP = (-1024.0, 3071.0)
STRUCT3 = np.ones((3, 3, 3), bool)
STRUCT2 = np.ones((3, 3), bool)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--aorta-mask", required=True, dest="mask")
    p.add_argument("--output", required=True)
    p.add_argument("--viz", default=None, help="write diagnostic PNG")
    p.add_argument("--features", default=None, help="append per-candidate features to CSV")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--profile", choices=("spec", "loose", "major"), default="spec")

    g = p.add_argument_group("ROI")
    g.add_argument("--margin-mm", type=float, default=30.0)

    g = p.add_argument_group("intensity")
    g.add_argument("--core-erode-mm", type=float, default=2.0)
    g.add_argument("--lumen-pct", type=float, default=85.0)
    g.add_argument("--thr-frac", type=float, default=0.50)
    g.add_argument("--ceiling-frac", type=float, default=1.60)
    g.add_argument("--hu-ceiling", type=float, default=None)
    g.add_argument("--thr-max", type=float, default=None)
    g.add_argument("--hu-ceiling-max", type=float, default=600.0)

    g = p.add_argument_group("candidate search")
    g.add_argument("--collar-mm", type=float, default=6.0)
    g.add_argument("--rind-mm", type=float, default=0.8)
    g.add_argument("--touch-mm", type=float, default=1.5)
    g.add_argument("--min-cand-voxels", type=int, default=6)
    g.add_argument("--cap-margin-mm", type=float, default=5.0)
    g.add_argument("--cap-cos", type=float, default=0.85)

    g = p.add_argument_group("eligibility / origin geometry")
    g.add_argument("--grow-mm", type=float, default=14.0)
    g.add_argument("--min-reach-mm", type=float, default=5.0,
                   help="lumen must remain followable for this path length beyond the wall")
    g.add_argument("--min-ostium-mm", type=float, default=2.0)
    g.add_argument("--min-branch-angle-deg", type=float, default=15.0,
                   help="minimum acute angle between daughter direction and local aortic axis")
    g.add_argument("--min-outward-gain-mm", type=float, default=0.8,
                   help="minimum increase in 3-D distance-to-aorta over the first 5 mm path")
    g.add_argument("--min-outward-fraction", type=float, default=0.45,
                   help="fraction of early path steps that must not move back toward the aorta")

    g = p.add_argument_group("shape / intensity")
    g.add_argument("--min-anisotropy", type=float, default=1.10)
    g.add_argument("--min-elongation", type=float, default=1.3)
    g.add_argument("--max-leak", type=float, default=30.0)
    g.add_argument("--min-caliber-mm", type=float, default=1.10)
    g.add_argument("--max-caliber-mm", type=float, default=10.0)
    g.add_argument("--max-ostium-mm", type=float, default=12.0)
    g.add_argument("--max-bright-ratio", type=float, default=1.05)
    g.add_argument("--min-bright-ratio", type=float, default=0.35)

    g = p.add_argument_group("measurement")
    g.add_argument("--trace-mm", type=float, default=10.0)
    g.add_argument("--seed-mm", type=float, default=5.0)
    g.add_argument("--dir-fit-mm", type=float, default=3.0)
    g.add_argument("--min-radius-mm", type=float, default=0.4)
    g.add_argument("--min-seed-radius-mm", type=float, default=None)
    g.add_argument("--merge-mm", type=float, default=2.5)
    g.add_argument("--ostium-push-mm", type=float, default=0.25)
    g.add_argument("--ostium-mode", choices=("snap", "centroid", "weighted"), default="weighted")
    g.add_argument("--bifurcation-shell-mm", type=float, default=1.5,
                   help="shell thickness used for conservative first-bifurcation detection")

    a = p.parse_args(argv)
    given = {tok.split("=")[0] for tok in (argv if argv is not None else sys.argv[1:]) if tok.startswith("--")}
    presets = {
        "spec": {"--min-ostium-mm": 2.0, "--min-seed-radius-mm": 1.0},
        "loose": {"--min-ostium-mm": 1.0, "--min-seed-radius-mm": 0.0},
        "major": {"--min-ostium-mm": 2.5, "--min-seed-radius-mm": 1.5},
    }
    for flag, val in presets[a.profile].items():
        if flag not in given:
            setattr(a, flag[2:].replace("-", "_"), val)
    if a.min_seed_radius_mm is None:
        a.min_seed_radius_mm = 0.0
    return a


def log(on, *a):
    if on:
        print("  ", *a, file=sys.stderr)


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def equiv_diam(n_vox, area_per_vox):
    return 2.0 * np.sqrt(max(float(n_vox), 1.0) * float(area_per_vox) / np.pi)


def _read_via_nibabel(path, label=None):
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError(f"cannot read {path}: install nibabel to handle non-standard headers")
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    aff = np.asarray(nii.affine, float)
    m_lps = np.diag([-1.0, -1.0, 1.0]) @ aff[:3, :3]
    spacing = np.linalg.norm(m_lps, axis=0)
    spacing[spacing < 1e-9] = 1.0
    direction = m_lps / spacing
    u, _, vt = np.linalg.svd(direction)
    ortho = u @ vt
    skew = float(np.abs(ortho - direction).max())
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    img.SetSpacing([float(v) for v in spacing])
    img.SetOrigin([float(-aff[0, 3]), float(-aff[1, 3]), float(aff[2, 3])])
    img.SetDirection([float(v) for v in ortho.flatten()])
    print(f"note: {label or os.path.basename(path)} non-orthonormal direction; orthonormalised (max {skew:.2e})",
          file=sys.stderr)
    if skew > 0.01:
        print("WARNING: large header-direction correction; physical coordinates may be less reliable", file=sys.stderr)
    return img


def read_image_any(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        return sitk.ReadImage(path)
    except RuntimeError:
        with open(path, "rb") as f:
            magic = f.read(2)
        if magic != b"\x1f\x8b":
            return _read_via_nibabel(path)
        tmpdir = tempfile.mkdtemp(prefix="branchseed_")
        alias = os.path.join(tmpdir, os.path.basename(path) + ".gz")
        try:
            try:
                os.symlink(os.path.abspath(path), alias)
            except (OSError, NotImplementedError, AttributeError):
                shutil.copyfile(path, alias)
            try:
                img = sitk.ReadImage(alias)
            except RuntimeError:
                img = _read_via_nibabel(alias, os.path.basename(path))
            print(f"note: {os.path.basename(path)} is gzip-compressed despite its .nii name", file=sys.stderr)
            return img
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def load_and_crop(image_path, mask_path, margin_mm, debug=False):
    img = read_image_any(image_path)
    msk = read_image_any(mask_path)
    if img.GetSize() != msk.GetSize():
        raise ValueError(f"grid mismatch: image {img.GetSize()} vs mask {msk.GetSize()}")
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    mask = sitk.GetArrayFromImage(msk) > 0
    if not mask.any():
        raise ValueError("aorta mask is empty")
    spacing = np.array(list(reversed(img.GetSpacing())), float)
    orig_shape = np.array(mask.shape)
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    caps = {(ax, end): bool((lo[ax] == 0) if end == 0 else (hi[ax] == orig_shape[ax]))
            for ax in range(3) for end in (0, 1)}
    pad = np.ceil(margin_mm / spacing).astype(int)
    lo_p = np.maximum(lo - pad, 0)
    hi_p = np.minimum(hi + pad, orig_shape)
    sl = tuple(slice(a, b) for a, b in zip(lo_p, hi_p))
    vol_c = np.ascontiguousarray(vol[sl])
    np.clip(vol_c, *HU_CLIP, out=vol_c)
    mask_c = np.ascontiguousarray(mask[sl])
    log(debug, f"grid {img.GetSize()} spacing {tuple(round(s,3) for s in img.GetSpacing())}")
    log(debug, f"ROI {vol_c.shape} = {100*vol_c.size/np.prod(orig_shape):.2f}% of volume")
    log(debug, f"mask on volume boundary: {[k for k,v in caps.items() if v]}")
    return dict(img=img, vol=vol_c, mask=mask_c, spacing=spacing, offset=lo_p, caps=caps)


def to_physical(case, idx_zyx):
    z, y, x = np.asarray(idx_zyx, float) + case["offset"]
    return np.asarray(case["img"].TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))), float)


def intensity_model(case, args):
    vol, mask, sp = case["vol"], case["mask"], case["spacing"]
    d_in = ndimage.distance_transform_edt(mask, sampling=sp)
    core = d_in > args.core_erode_mm
    if core.sum() < 50:
        core = mask
    lumen = float(np.percentile(vol[core], args.lumen_pct))
    soft_band = vol[(vol > -20) & (vol < 120)]
    soft = float(np.median(soft_band)) if soft_band.size > 500 else 40.0
    thr = soft + args.thr_frac * (lumen - soft)
    if args.thr_max is not None:
        thr = min(thr, args.thr_max)
    ceiling = args.hu_ceiling if args.hu_ceiling is not None else soft + args.ceiling_frac * (lumen - soft)
    ceiling = min(float(ceiling), args.hu_ceiling_max)
    sd = float(np.std(vol[core]))
    med = float(np.median(vol[core]))
    log(args.debug, f"lumen {lumen:.0f} HU, soft {soft:.0f} HU -> band {thr:.0f} .. {ceiling:.0f} HU")
    if lumen - soft < 80:
        print("WARNING: weak arterial enhancement", file=sys.stderr)
    if sd > 0.22 * max(lumen - soft, 1.0):
        print(f"WARNING: heterogeneous aorta mask interior (median {med:.0f}, p{args.lumen_pct:.0f} {lumen:.0f}, sd {sd:.0f})",
              file=sys.stderr)
    case.update(d_in=d_in, lumen_hu=lumen, soft_hu=soft, thr=thr, hu_ceiling=ceiling)
    return case


def aorta_geometry(case, args):
    mask, sp = case["mask"], case["spacing"]
    d_out, near_idx = ndimage.distance_transform_edt(~mask, sampling=sp, return_indices=True)
    zs = np.flatnonzero(mask.any(axis=(1, 2)))
    cy = np.full(mask.shape[0], np.nan)
    cx = np.full(mask.shape[0], np.nan)
    for z in zs:
        yy, xx = np.nonzero(mask[z])
        cy[z], cx[z] = yy.mean(), xx.mean()
    if len(zs):
        good = ~np.isnan(cy)
        k = max(1, int(round(6.0 / sp[0])) | 1)
        k = min(k, int(good.sum()))
        if k > 1:
            ker = np.ones(k) / k
            cy[good] = np.convolve(cy[good], ker, mode="same")
            cx[good] = np.convolve(cx[good], ker, mode="same")
    tangent = np.zeros((mask.shape[0], 3), float)
    if len(zs):
        for z in zs:
            z0, z1 = max(z - 2, zs[0]), min(z + 2, zs[-1])
            tangent[z] = unit([(z1-z0)*sp[0], (cy[z1]-cy[z0])*sp[1], (cx[z1]-cx[z0])*sp[2]])
    surface = (d_out > 0) & (d_out <= float(sp.min()) * 1.05)
    cap_zone = np.zeros_like(mask)
    for (ax, end), touched in case["caps"].items():
        if not touched:
            continue
        w = int(np.ceil(args.cap_margin_mm / sp[ax]))
        sl = [slice(None)] * 3
        sl[ax] = slice(0, min(w+1, mask.shape[ax])) if end == 0 else slice(max(mask.shape[ax]-w-1, 0), mask.shape[ax])
        cap_zone[tuple(sl)] = True
    gz, gy, gx = np.gradient(d_out, *sp)
    nrm = np.sqrt(gz*gz + gy*gy + gx*gx) + 1e-9
    cos_ax = np.abs(gz*tangent[:,0,None,None] + gy*tangent[:,1,None,None] + gx*tangent[:,2,None,None]) / nrm
    cap_zone |= (cos_ax > args.cap_cos) & surface
    searchable = surface & ~cap_zone
    log(args.debug, f"wall {surface.sum()} voxels; caps remove {(surface&cap_zone).sum()}; searchable {searchable.sum()}")
    case.update(d_out=d_out, near_idx=near_idx, surface=surface, searchable=searchable,
                cap_zone=cap_zone, axis_cy=cy, axis_cx=cx, aorta_tangent_zyx=tangent)
    return case


def local_aorta_axis_physical(case, ostium_idx):
    z = int(np.clip(round(float(ostium_idx[0])), 0, len(case["aorta_tangent_zyx"])-1))
    t = case["aorta_tangent_zyx"][z]
    axis_xyz = np.array([t[2], t[1], t[0]], float)
    D = np.asarray(case["img"].GetDirection(), float).reshape(3, 3)
    return unit(D @ axis_xyz)


def find_candidates(case, args):
    vol, d = case["vol"], case["d_out"]
    bright = (vol >= case["thr"]) & (vol <= case["hu_ceiling"])
    collar = (~case["mask"]) & (d > args.rind_mm) & (d <= args.collar_mm)
    seedable = collar & bright & ~case["cap_zone"]
    lab, n = ndimage.label(seedable, structure=STRUCT3)
    if n == 0:
        return [], lab
    near_wall = (~case["mask"]) & (d <= args.rind_mm + args.touch_mm) & ~case["cap_zone"]
    touching = set(np.unique(lab[near_wall & seedable])) - {0}
    sizes = ndimage.sum(seedable, lab, index=np.arange(1, n+1))
    cands = [int(i) for i in sorted(touching) if sizes[i-1] >= args.min_cand_voxels]
    log(args.debug, f"{n} collar components, {len(touching)} touch wall, {len(cands)} pass size")
    return cands, lab


def grow(case, lab, li, args):
    vol, mask, sp, d = case["vol"], case["mask"], case["spacing"], case["d_out"]
    seed_region = lab == li
    bright = (vol >= case["thr"]) & (vol <= case["hu_ceiling"]) & ~mask
    reachable = bright & (d > args.rind_mm) & (d <= args.grow_mm)
    start = np.argwhere(seed_region & (d <= args.rind_mm + args.touch_mm))
    if start.size == 0:
        start = np.argwhere(seed_region)
    if start.size == 0:
        return None, "no start voxels"
    mcp = MCP_Geometric(np.where(reachable, 1.0, np.inf), sampling=tuple(sp))
    gdist, _ = mcp.find_costs([tuple(s) for s in start])
    grown = np.isfinite(gdist) & (gdist <= args.grow_mm) & reachable
    if grown.sum() < args.min_cand_voxels:
        return None, "too small after growth"
    prox = grown & (gdist <= args.min_reach_mm)
    p75 = float(np.percentile(vol[prox], 75)) if prox.any() else -1000.0
    ratio = (p75 - case["soft_hu"]) / max(case["lumen_hu"] - case["soft_hu"], 1.0)
    return dict(label=li, grown=grown, gdist=gdist, mcp=mcp,
                raw_reach=float(gdist[grown].max()),
                grown_mm3=float(grown.sum())*float(np.prod(sp)), bright_ratio=ratio), None


def place_ostium(case, c, args):
    sp, d, near = case["spacing"], case["d_out"], case["near_idx"]
    band = float(args.rind_mm + 1.5 * sp.max())
    prox = c["grown"] & (d <= band)
    if not prox.any():
        dmin = float(d[c["grown"]].min())
        prox = c["grown"] & (d <= dmin + 1.5*sp.max())
    if not prox.any():
        return None, "no proximal segment"
    allfeet = np.stack([near[k][prox] for k in range(3)], axis=1)
    feet, counts = np.unique(allfeet, axis=0, return_counts=True)
    if args.ostium_mode == "weighted":
        centre = (feet * counts[:,None]).sum(axis=0) / counts.sum()
    else:
        centre = feet.mean(axis=0)
    if args.ostium_mode == "snap":
        d2 = (((feet-centre)*sp)**2).sum(axis=1)
        centre = feet[int(np.argmin(d2))].astype(float)
    centre = np.asarray(centre, float)
    if args.ostium_push_mm > 0:
        z = int(np.clip(round(centre[0]), 0, len(case["axis_cy"])-1))
        ay, ax = case["axis_cy"][z], case["axis_cx"][z]
        if np.isfinite(ay) and np.isfinite(ax):
            radial_mm = np.array([0.0, (centre[1]-ay)*sp[1], (centre[2]-ax)*sp[2]])
            nr = np.linalg.norm(radial_mm)
            if nr > 1e-6:
                centre = centre + (radial_mm/nr) * args.ostium_push_mm / sp
    thickness = max(band-args.rind_mm, float(sp.min()))
    area = float(prox.sum()) * float(np.prod(sp)) / thickness
    diam = float(2*np.sqrt(max(area, 1e-6)/np.pi))
    if diam < args.min_ostium_mm:
        return None, f"ostium {diam:.1f} < {args.min_ostium_mm} mm"
    if diam > args.max_ostium_mm:
        return None, f"ostium {diam:.1f} > {args.max_ostium_mm} mm"
    c.update(ostium_idx=centre, patch_pts=feet, patch_counts=counts,
             proximal_mask=prox, ostium_diam_mm=diam)
    return c, None


def merge_trunks(case, cands, args):
    sp = case["spacing"]
    keep, dropped = [], set()
    wall_adj = max(1.25, float(sp.max())*1.25)
    for i, a in enumerate(cands):
        if i in dropped:
            continue
        for j in range(i+1, len(cands)):
            if j in dropped:
                continue
            b = cands[j]
            od = float(np.linalg.norm((a["ostium_idx"]-b["ostium_idx"])*sp))
            if od > args.merge_mm:
                continue
            pa = a["grown"] & (a["gdist"] <= args.min_reach_mm)
            pb = b["grown"] & (b["gdist"] <= args.min_reach_mm)
            prox_overlap = bool((pa & pb).any())
            dpatch = np.linalg.norm((a["patch_pts"][:,None,:]-b["patch_pts"][None,:,:])*sp, axis=2)
            same_wall_patch = bool(dpatch.min() <= wall_adj)
            if prox_overlap and same_wall_patch:
                score_a = int(pa.sum())
                score_b = int(pb.sum())
                if score_b > score_a:
                    a = b
                dropped.add(j)
        keep.append(a)
    return keep


def orient_path_from_ostium(case, path, ostium_idx):
    if len(path) < 2:
        return path
    ost = to_physical(case, ostium_idx)
    d0 = np.linalg.norm(to_physical(case, path[0]) - ost)
    d1 = np.linalg.norm(to_physical(case, path[-1]) - ost)
    return path if d0 <= d1 else path[::-1].copy()


def interpolate_on_path(path, arc, target):
    target = float(np.clip(target, arc[0], arc[-1]))
    j = int(np.searchsorted(arc, target, side="right"))
    if j <= 0:
        return path[0].astype(float)
    if j >= len(path):
        return path[-1].astype(float)
    a0, a1 = arc[j-1], arc[j]
    if a1 <= a0 + 1e-9:
        return path[j].astype(float)
    w = (target-a0)/(a1-a0)
    return (1-w)*path[j-1] + w*path[j]


def triplanar_signature(region, point, spacing):
    zi, yi, xi = [int(np.clip(round(v), 0, region.shape[k]-1)) for k,v in enumerate(point)]
    planes = [
        (0, zi, (yi,xi), spacing[1]*spacing[2]),
        (1, yi, (zi,xi), spacing[0]*spacing[2]),
        (2, xi, (zi,yi), spacing[0]*spacing[1]),
    ]
    out = []
    for ax, idx, (a,b), area in planes:
        plane = region.take(idx, axis=ax)
        if not plane[a,b]:
            return None, None
        lab, _ = ndimage.label(plane, structure=STRUCT2)
        li = lab[a,b]
        if li == 0:
            return None, None
        out.append((equiv_diam(int((lab==li).sum()), area), ax))
    out.sort()
    return [float(d) for d,_ in out], out[0][1]


def recentre(field, point, spacing, reach_mm):
    p = np.array([int(np.clip(round(v), 0, field.shape[k]-1)) for k,v in enumerate(point)])
    w = np.maximum(np.ceil(reach_mm/spacing).astype(int), 1)
    sl = tuple(slice(max(p[k]-w[k],0), min(p[k]+w[k]+1,field.shape[k])) for k in range(3))
    sub = field[sl]
    if sub.size == 0 or sub.max() <= 0:
        return p.astype(float)
    zz,yy,xx = np.mgrid[sl[0],sl[1],sl[2]]
    dist = np.sqrt(((zz-p[0])*spacing[0])**2 + ((yy-p[1])*spacing[1])**2 + ((xx-p[2])*spacing[2])**2)
    cand = np.where(dist <= reach_mm, sub, -1.0)
    off = np.unravel_index(int(np.argmax(cand)), cand.shape)
    return np.array([sl[k].start+off[k] for k in range(3)], float)


def path_distance_from_aorta(case, path):
    pts = np.rint(path).astype(int)
    for k in range(3):
        pts[:,k] = np.clip(pts[:,k], 0, case["d_out"].shape[k]-1)
    return case["d_out"][pts[:,0], pts[:,1], pts[:,2]].astype(float)


def conservative_bifurcation_arc(case, c, path, arc, args):
    if arc[-1] < 4.0:
        return None
    sp = case["spacing"]
    shell = max(args.bifurcation_shell_mm, float(sp.min()))
    hits = []
    for t in np.arange(3.0, min(args.trace_mm, arc[-1]-1.0), 1.0):
        p = interpolate_on_path(path, arc, t)
        rad = np.maximum(np.ceil(5.0 / sp).astype(int), 1)
        ctr = np.rint(p).astype(int)
        lo = np.maximum(ctr-rad, 0)
        hi = np.minimum(ctr+rad+1, np.array(c["grown"].shape))
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        zz,yy,xx = np.mgrid[sl[0], sl[1], sl[2]]
        r2 = ((zz-p[0])*sp[0])**2 + ((yy-p[1])*sp[1])**2 + ((xx-p[2])*sp[2])**2
        local = r2 <= 5.0**2
        sh = c["grown"][sl] & local & (c["gdist"][sl] >= t+0.5) & (c["gdist"][sl] <= t+0.5+shell)
        lab, n = ndimage.label(sh, structure=STRUCT3)
        sizes = ndimage.sum(sh, lab, index=np.arange(1,n+1)) if n else []
        substantial = sum(float(s) >= 3 for s in sizes)
        hits.append((t, substantial >= 2))
    for k in range(len(hits)-1):
        if hits[k][1] and hits[k+1][1]:
            return float(hits[k][0])
    return None


def trace_from_wall(case, c, args):
    window = c["grown"] & (c["gdist"] <= max(args.trace_mm, args.seed_mm) + 2.0)
    if not window.any():
        return None, "empty trace window"
    candidates = np.argwhere(window)
    far = candidates[int(np.argmax(c["gdist"][window]))]
    try:
        raw = np.asarray(c["mcp"].traceback(tuple(far)), float)
    except Exception:
        return None, "traceback failed"
    if len(raw) < 2:
        return None, "path too short"
    path = orient_path_from_ostium(case, raw, c["ostium_idx"])
    if np.linalg.norm((path[0]-c["ostium_idx"])*case["spacing"]) > 0.1:
        path = np.vstack([c["ostium_idx"], path])
    phys = np.asarray([to_physical(case, p) for p in path])
    steps = np.linalg.norm(np.diff(phys, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] < args.min_reach_mm:
        return None, f"wall-to-lumen reach {arc[-1]:.1f} < {args.min_reach_mm} mm"
    upto = arc <= min(args.seed_mm, arc[-1]) + 1e-6
    p5 = path[upto]
    if len(p5) < 2:
        return None, "not enough early path samples"
    dout = path_distance_from_aorta(case, p5)
    gain = float(dout[-1] - dout[0])
    increments = np.diff(dout)
    frac = float(np.mean(increments >= -0.25*case["spacing"].min())) if len(increments) else 0.0
    if gain < args.min_outward_gain_mm:
        return None, f"outward gain {gain:.2f} < {args.min_outward_gain_mm} mm"
    if frac < args.min_outward_fraction:
        return None, f"outward fraction {frac:.2f} < {args.min_outward_fraction}"
    seed_raw = interpolate_on_path(path, arc, min(args.seed_mm, arc[-1]))
    c.update(path=path, arc=arc, path_phys=phys, outward_gain_mm=gain,
             outward_fraction=frac, seed_raw_idx=seed_raw,
             wall_reach_mm=float(arc[-1]))
    bif = conservative_bifurcation_arc(case, c, path, arc, args)
    c["bifurcation_mm"] = bif
    c["trace_end_mm"] = min(args.trace_mm, arc[-1], bif if bif is not None else np.inf)
    return c, None


def measure(case, c, args):
    sp, grown = case["spacing"], c["grown"]
    d_branch = ndimage.distance_transform_edt(grown, sampling=sp)
    probe, _ = triplanar_signature(grown, c["seed_raw_idx"], sp)
    recenter_mm = float(np.clip(0.6*probe[0], 1.0, 4.0)) if probe else 1.0
    seed_idx = recentre(d_branch, c["seed_raw_idx"], sp, recenter_mm)
    dims, axis_min = triplanar_signature(grown, seed_idx, sp)
    if dims is None:
        return None, "signature undefined at seed"
    d_min, d_mid, d_max = dims
    anis = d_max / max(d_min, 1e-6)
    elong = c["wall_reach_mm"] / max(d_min, 1e-6)
    c.update(d_min=d_min, d_mid=d_mid, d_max=d_max, anisotropy=anis, elongation=elong, axis_min=axis_min)
    if d_min < args.min_caliber_mm:
        return None, f"caliber {d_min:.1f} < {args.min_caliber_mm} mm"
    if d_min > args.max_caliber_mm:
        return None, f"caliber {d_min:.1f} > {args.max_caliber_mm} mm"
    if anis < args.min_anisotropy:
        return None, f"anisotropy {anis:.2f} < {args.min_anisotropy}"
    if elong < args.min_elongation:
        return None, f"elongation {elong:.2f} < {args.min_elongation}"
    if c["bright_ratio"] < args.min_bright_ratio:
        return None, f"bright ratio {c['bright_ratio']:.2f} < {args.min_bright_ratio}"
    if c["bright_ratio"] > args.max_bright_ratio:
        return None, f"bright ratio {c['bright_ratio']:.2f} > {args.max_bright_ratio}"
    tube_mm3 = np.pi*(d_min/2.0)**2 * max(c["wall_reach_mm"], 1e-3)
    leak = float(c["grown_mm3"] / max(tube_mm3, 1e-6))
    c["leak"] = leak
    if leak > args.max_leak:
        return None, f"leak {leak:.1f} > {args.max_leak}"
    zi,yi,xi = [int(np.clip(round(v), 0, grown.shape[k]-1)) for k,v in enumerate(seed_idx)]
    edt_radius = float(d_branch[zi,yi,xi])
    local_radius = min(edt_radius, 0.65*d_min)
    if local_radius < args.min_seed_radius_mm:
        return None, f"seed radius {local_radius:.2f} < {args.min_seed_radius_mm} mm"
    radius = max(local_radius, args.min_radius_mm)
    ost_mm = to_physical(case, c["ostium_idx"])
    seed_mm = to_physical(case, seed_idx)
    chord = seed_mm - ost_mm
    direction = unit(chord)
    if np.linalg.norm(chord) < 0.5*args.seed_mm:
        head = c["path_phys"][c["arc"] <= max(args.dir_fit_mm, 2.0)]
        if len(head) >= 2:
            _,_,vt = np.linalg.svd(head-head.mean(axis=0), full_matrices=False)
            direction = unit(vt[0])
            if np.dot(direction, chord) < 0:
                direction = -direction
    aorta_axis = local_aorta_axis_physical(case, c["ostium_idx"])
    dot = float(np.clip(abs(np.dot(direction, aorta_axis)), 0.0, 1.0))
    branch_angle = float(np.degrees(np.arccos(dot)))
    if branch_angle < args.min_branch_angle_deg:
        return None, f"branch angle {branch_angle:.1f} < {args.min_branch_angle_deg} deg"
    phys_axis = {0:2, 1:1, 2:0}
    axis_dot = float(abs(direction[phys_axis[axis_min]]))
    c.update(seed_idx=seed_idx, ostium_mm=ost_mm, seed_mm=seed_mm, radius_mm=radius,
             direction=direction, branch_angle_deg=branch_angle, axis_dot=axis_dot)
    return c, None


def emit(case_id, cands, out_path):
    ordered = sorted(cands, key=lambda c: -c["ostium_mm"][2])
    daughters = []
    for n,c in enumerate(ordered,1):
        daughters.append({
            "instance_id": f"branch_{n:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(float(v),3) for v in c["ostium_mm"]],
            "seed_xyz_mm": [round(float(v),3) for v in c["seed_mm"]],
            "radius_mm": round(float(c["radius_mm"]),3),
            "direction_xyz": [round(float(v),5) for v in c["direction"]],
        })
    payload = {"case_id": case_id, "parent":{"instance_id":"aorta"}, "daughters":daughters}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path,"w") as f:
        json.dump(payload,f,indent=2)
    return payload


def write_features(path, case_id, rows):
    cols = ["case_id","accepted","reject","ostium_diam_mm","wall_reach_mm","outward_gain_mm",
            "outward_fraction","branch_angle_deg","bifurcation_mm","d_min_mm","d_mid_mm","d_max_mm",
            "anisotropy","elongation","leak","grown_mm3","bright_ratio","radius_mm","axis_dot",
            "ostium_x","ostium_y","ostium_z"]
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            r = dict(r); r["case_id"] = case_id; w.writerow(r)


def _overlay_slice(ax, img2, mask2, det2, ost_xy, seed_xy, title, vmin, vmax, aspect=1.0):
    ax.imshow(img2, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
    if mask2.any():
        ax.contour(mask2.astype(float), levels=[0.5], colors="#3fa7c4", linewidths=0.8)
    if det2.any():
        ax.contour(det2.astype(float), levels=[0.5], colors="#ff9500", linewidths=0.7)
    ax.plot([ost_xy[0]],[ost_xy[1]],"o",ms=6,mfc="none",mec="#e8443f",mew=1.4)
    ax.annotate("",xy=seed_xy,xytext=ost_xy,arrowprops=dict(arrowstyle="->",color="#e8443f",lw=1.2))
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def visual_check(case, cands, png_path, max_rows=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vol, mask, sp = case["vol"], case["mask"], case["spacing"]
    shown = cands[:max_rows]
    if not shown:
        fig,ax = plt.subplots(1,1,figsize=(6,4)); ax.text(.5,.5,"No accepted daughters",ha="center",va="center"); ax.axis("off")
    else:
        fig,axes = plt.subplots(len(shown),3,figsize=(12,3.2*len(shown)), squeeze=False)
        for r,c in enumerate(shown):
            z,y,x = [int(np.clip(round(v),0,vol.shape[k]-1)) for k,v in enumerate(c["seed_idx"])]
            oz,oy,ox = c["ostium_idx"]; sz,sy,sx = c["seed_idx"]
            det = c["grown"]
            _overlay_slice(axes[r,0], vol[z], mask[z], det[z], (ox,oy), (sx,sy),
                           f"branch {r+1} axial  angle={c['branch_angle_deg']:.1f}°", case["soft_hu"]-100, case["lumen_hu"]*1.05, sp[1]/sp[2])
            _overlay_slice(axes[r,1], vol[:,y,:], mask[:,y,:], det[:,y,:], (ox,oz), (sx,sz),
                           f"coronal  r={c['radius_mm']:.2f}mm", case["soft_hu"]-100, case["lumen_hu"]*1.05, sp[0]/sp[2])
            _overlay_slice(axes[r,2], vol[:,:,x], mask[:,:,x], det[:,:,x], (oy,oz), (sy,sz),
                           f"sagittal  outward={c['outward_gain_mm']:.2f}mm", case["soft_hu"]-100, case["lumen_hu"]*1.05, sp[0]/sp[1])
        extra = len(cands)-len(shown)
        fig.suptitle(f"{os.path.basename(png_path)} — exact seed slices — {len(cands)} daughters" + (f" (showing first {len(shown)})" if extra>0 else ""), fontsize=10)
        fig.tight_layout(rect=(0,0,1,0.98))
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path,dpi=120,bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    case = load_and_crop(args.image, args.mask, args.margin_mm, args.debug)
    case = intensity_model(case, args)
    case = aorta_geometry(case, args)
    labels, lab = find_candidates(case, args)
    rows, stage, rejects = [], [], []
    for li in labels:
        c,why = grow(case,lab,li,args)
        if c is None:
            rejects.append(why); rows.append(dict(accepted=0,reject=why)); continue
        c,why = place_ostium(case,c,args)
        if c is None:
            rejects.append(why); rows.append(dict(accepted=0,reject=why)); continue
        stage.append(c)
    stage = merge_trunks(case, stage, args)
    final = []
    for c in stage:
        c,why = trace_from_wall(case,c,args)
        if c is not None:
            c,why = measure(case,c,args)
        row = dict(accepted=0, reject=why or "",
                   ostium_diam_mm=round(c["ostium_diam_mm"],3) if c is not None and "ostium_diam_mm" in c else "",
                   wall_reach_mm=round(c.get("wall_reach_mm",0),3) if c is not None else "",
                   outward_gain_mm=round(c.get("outward_gain_mm",0),3) if c is not None else "",
                   outward_fraction=round(c.get("outward_fraction",0),3) if c is not None else "",
                   bifurcation_mm=round(c["bifurcation_mm"],3) if c is not None and c.get("bifurcation_mm") is not None else "",
                   grown_mm3=round(c.get("grown_mm3",0),1) if c is not None else "",
                   bright_ratio=round(c.get("bright_ratio",0),3) if c is not None else "")
        if c is None:
            rejects.append(why); rows.append(row); continue
        for k in ("d_min","d_mid","d_max","anisotropy","elongation","leak","branch_angle_deg"):
            if k in c:
                row[k + ("_mm" if k.startswith("d_") else "")] = round(c[k],3)
        row.update(accepted=1,reject="",radius_mm=round(c["radius_mm"],3),axis_dot=round(c["axis_dot"],3),
                   ostium_x=round(c["ostium_mm"][0],2),ostium_y=round(c["ostium_mm"][1],2),ostium_z=round(c["ostium_mm"][2],2))
        rows.append(row); final.append(c)
    case_id = os.path.basename(os.path.dirname(os.path.abspath(args.image))) or "case"
    payload = emit(case_id, final, args.output)
    if args.viz:
        visual_check(case, final, args.viz)
    if args.features:
        write_features(args.features, case_id, rows)
    if args.debug:
        if rejects:
            log(True,"rejections:",dict(Counter(r.split("(")[0].split("<")[0].strip() for r in rejects)))
        for c in final:
            log(True, f"d=({c['d_min']:.1f},{c['d_mid']:.1f},{c['d_max']:.1f})mm "
                      f"r={c['radius_mm']:.2f} angle={c['branch_angle_deg']:.1f}deg "
                      f"out={c['outward_gain_mm']:.2f}mm frac={c['outward_fraction']:.2f} "
                      f"bif={c['bifurcation_mm'] if c['bifurcation_mm'] is not None else '-'}")
    print(f"{case_id}: {len(payload['daughters'])} daughters ({len(labels)} candidates, {len(rejects)} rejected) in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
