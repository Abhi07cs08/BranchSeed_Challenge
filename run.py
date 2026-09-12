#!/usr/bin/env python3
"""Branchseed implementation v2: mask-anchored direct-aortic-daughter detector."""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

try:
    from skimage.graph import MCP_Geometric
except ImportError:
    sys.exit("need scikit-image: pip install scikit-image")

STRUCT3 = np.ones((3, 3, 3), bool)
STRUCT2 = np.ones((3, 3), bool)


def args_parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--aorta-mask", required=True, dest="mask")
    p.add_argument("--output", required=True)
    p.add_argument("--viz")
    p.add_argument("--features")
    p.add_argument("--debug", action="store_true")
    for name, default in [
        ("margin-mm", 30.0),
        ("core-erode-mm", 2.0),
        ("thr-frac", 0.40),
        ("ceiling-frac", 1.60),
        ("touch-mm", 1.6),
        ("distal-shell-width-mm", 2.0),
        ("wall-assoc-mm", 3.0),
        ("cap-margin-mm", 3.0),
        ("cap-cos", 0.85),
        ("grow-mm", 22.0),
        ("min-reach-mm", 5.0),
        ("min-radial-reach-mm", 5.0),
        ("min-outward-progress-frac", 0.55),
        ("min-outward-dot", 0.10),
        ("min-ostium-mm", 1.0),
        ("min-branch-angle-deg", 20.0),
        ("min-anisotropy", 1.35),
        ("min-elongation", 1.8),
        ("min-caliber-mm", 1.10),
        ("max-caliber-mm", 14.0),
        ("min-bright-ratio", 0.35),
        ("trace-mm", 10.0),
        ("seed-mm", 5.0),
        ("min-radius-mm", 0.4),
        ("bifurcation-step-mm", 0.75),
        ("dedup-mm", 1.5),
    ]:
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--hu-ceiling", type=float)
    p.add_argument("--min-wall-patch-voxels", type=int, default=2)
    p.add_argument("--min-cand-voxels", type=int, default=6)
    p.add_argument("--min-distal-voxels", type=int, default=3)
    return p.parse_args(argv)


def log(on, *x):
    if on:
        print(*x, file=sys.stderr)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def to_phys(c, p):
    z, y, x = np.asarray(p, float) + c["offset"]
    return np.asarray(
        c["img"].TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
    )


def load_case(image, mask, margin, debug=False):
    img = sitk.ReadImage(image)
    m = sitk.ReadImage(mask)
    if (
        img.GetSize() != m.GetSize()
        or img.GetSpacing() != m.GetSpacing()
        or img.GetOrigin() != m.GetOrigin()
        or img.GetDirection() != m.GetDirection()
    ):
        raise ValueError("image and mask must use the same physical grid")
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    mk = sitk.GetArrayFromImage(m) > 0
    if not mk.any():
        raise ValueError("aorta mask is empty")
    sp = np.asarray(img.GetSpacing()[::-1], float)
    idx = np.argwhere(mk)
    lo = idx.min(0)
    hi = idx.max(0) + 1
    shape = np.asarray(mk.shape)
    caps = {
        (ax, e): (lo[ax] == 0 if e == 0 else hi[ax] == shape[ax])
        for ax in range(3)
        for e in (0, 1)
    }
    pad = np.ceil(margin / sp).astype(int)
    l = np.maximum(lo - pad, 0)
    h = np.minimum(hi + pad, shape)
    sl = tuple(slice(int(x), int(y)) for x, y in zip(l, h))
    v = np.clip(a[sl], -1024, 3071)
    mm = mk[sl]
    log(debug, "ROI", v.shape, "spacing zyx", sp)
    return dict(
        img=img,
        vol=np.ascontiguousarray(v),
        mask=np.ascontiguousarray(mm),
        spacing=sp,
        offset=l,
        caps=caps,
    )


def intensity(c, a):
    v, m, sp = c["vol"], c["mask"], c["spacing"]
    din = ndimage.distance_transform_edt(m, sampling=sp)
    core = din > a.core_erode_mm
    if core.sum() < 50:
        core = m
    lumen = float(np.median(v[core]))
    b = v[(v > -20) & (v < 120)]
    soft = float(np.median(b)) if b.size > 500 else 40.0
    thr = soft + a.thr_frac * (lumen - soft)
    ceil = (
        float(a.hu_ceiling)
        if a.hu_ceiling is not None
        else soft + a.ceiling_frac * (lumen - soft)
    )
    c.update(din=din, lumen_hu=lumen, soft_hu=soft, thr=thr, ceiling=ceil)
    log(a.debug, f"HU soft={soft:.0f} lumen={lumen:.0f} band={thr:.0f}..{ceil:.0f}")
    return c


def geometry(c, a):
    m, sp = c["mask"], c["spacing"]
    dout, near = ndimage.distance_transform_edt(~m, sampling=sp, return_indices=True)
    zs = np.flatnonzero(m.any((1, 2)))
    cy = np.zeros(m.shape[0])
    cx = np.zeros(m.shape[0])
    for z in zs:
        y, x = np.nonzero(m[z])
        cy[z] = y.mean()
        cx[z] = x.mean()
    if len(zs) > 1:
        yy = np.interp(np.arange(m.shape[0]), zs, cy[zs])
        xx = np.interp(np.arange(m.shape[0]), zs, cx[zs])
        yy = ndimage.gaussian_filter1d(yy, max(0.5, 3 / sp[0]))
        xx = ndimage.gaussian_filter1d(xx, max(0.5, 3 / sp[0]))
    else:
        yy, xx = cy, cx
    tang = np.zeros((m.shape[0], 3))
    for z in zs:
        z0 = max(int(z) - 2, int(zs[0]))
        z1 = min(int(z) + 2, int(zs[-1]))
        tang[z] = unit(
            [
                (z1 - z0) * sp[0],
                (yy[z1] - yy[z0]) * sp[1],
                (xx[z1] - xx[z0]) * sp[2],
            ]
        )
    surface = (dout > 0) & (dout <= sp.min() * 1.05)
    cap = np.zeros_like(m)
    for (ax, e), touch in c["caps"].items():
        if touch:
            w = int(np.ceil(a.cap_margin_mm / sp[ax]))
            sl = [slice(None)] * 3
            sl[ax] = slice(0, w + 1) if e == 0 else slice(max(m.shape[ax] - w - 1, 0), m.shape[ax])
            cap[tuple(sl)] = 1
    gz, gy, gx = np.gradient(dout, *sp)
    n = np.sqrt(gz * gz + gy * gy + gx * gx) + 1e-9
    ca = np.abs(
        gz * tang[:, 0, None, None]
        + gy * tang[:, 1, None, None]
        + gx * tang[:, 2, None, None]
    ) / n
    cap |= (ca > a.cap_cos) & surface
    cap_wall = np.zeros_like(m)
    cs = cap & surface
    if cs.any():
        f = np.stack([near[k][cs] for k in range(3)], 1)
        cap_wall[tuple(f.T)] = 1
    c.update(
        dout=dout,
        near=near,
        surface=surface,
        cap=cap,
        cap_wall=cap_wall,
        aorta_tangent_zyx=tang,
        aorta_center_y=yy,
        aorta_center_x=xx,
    )
    return c


def local_wall_normal(c, ostium_idx):
    """Approximate outward wall normal from local aortic centreline to ostium."""
    sp = c["spacing"]
    p = np.asarray(ostium_idx, float)
    z = int(np.clip(round(p[0]), 0, c["mask"].shape[0] - 1))
    centre = np.array([p[0], c["aorta_center_y"][z], c["aorta_center_x"][z]], float)
    radial = (p - centre) * sp
    tangent = unit(c["aorta_tangent_zyx"][z])
    radial = radial - np.dot(radial, tangent) * tangent
    return unit(radial)


def candidate_openings(c, a):
    """Find outward tubes first, then anchor each one back to the supplied mask wall.

    The distal shell at >= min_radial_reach_mm breaks the bright partial-volume rim
    around the aorta into separate outward continuations. Each continuation is then
    associated with a local wall patch, instead of labeling the whole bright wall rim
    as one candidate.
    """
    v, m, d, near, sp = c["vol"], c["mask"], c["dout"], c["near"], c["spacing"]
    bright = (v >= c["thr"]) & (v <= c["ceiling"])
    outside = bright & (~m) & (d > 0) & (d <= a.grow_mm)

    distal = outside & (d >= a.min_radial_reach_mm) & (
        d <= a.min_radial_reach_mm + a.distal_shell_width_mm
    )
    lab, n = ndimage.label(distal, STRUCT3)

    start_band = outside & (d <= max(a.touch_mm, sp.min() * 1.05))
    start_pts_all = np.argwhere(start_band)
    start_feet_all = (
        np.stack([near[k][start_band] for k in range(3)], 1)
        if len(start_pts_all)
        else np.empty((0, 3), int)
    )

    out = []
    for li in range(1, n + 1):
        distal_pts = np.argwhere(lab == li)
        if len(distal_pts) < a.min_distal_voxels:
            continue

        distal_mask = lab == li
        distal_feet = np.stack([near[k][distal_mask] for k in range(3)], 1)
        valid = ~c["cap_wall"][tuple(distal_feet.T)]
        distal_feet = distal_feet[valid]
        if not len(distal_feet):
            continue

        foot0 = np.median(distal_feet, axis=0)
        if len(start_pts_all):
            wall_dist = np.linalg.norm((start_feet_all - foot0) * sp, axis=1)
            use = wall_dist <= a.wall_assoc_mm
            starts = start_pts_all[use]
            feet = start_feet_all[use]
        else:
            starts = np.empty((0, 3), int)
            feet = np.empty((0, 3), int)

        if len(starts) < a.min_cand_voxels:
            continue

        feet_unique = np.unique(feet, axis=0)
        if len(feet_unique) < a.min_wall_patch_voxels:
            continue
        ostium = feet_unique.mean(axis=0)

        distal_centre = distal_pts.mean(axis=0)
        daughter_hint = unit((distal_centre - ostium) * sp)
        outward_dot = float(np.dot(local_wall_normal(c, ostium), daughter_hint))
        if outward_dot < a.min_outward_dot:
            continue

        out.append(
            dict(
                wall_id=li,
                wall_patch=feet_unique,
                start_pts=starts,
                distal_pts=distal_pts,
                distal_centre_idx=distal_centre,
                ostium_idx=ostium,
                outward_dot_hint=outward_dot,
            )
        )

    log(
        a.debug,
        f"outward-tube candidates: {len(out)} "
        f"(distal shell {a.min_radial_reach_mm:.1f}.."
        f"{a.min_radial_reach_mm + a.distal_shell_width_mm:.1f} mm from mask)",
    )
    return out


def grow(c, b, a):
    """Grow only through bright non-mask voxels, anchored to one local opening."""
    v, m, sp, d = c["vol"], c["mask"], c["spacing"], c["dout"]
    allowed = (
        (v >= c["thr"])
        & (v <= c["ceiling"])
        & (~m)
        & (d > 0)
        & (d <= a.grow_mm)
    )
    st = b["start_pts"]
    st = st[allowed[tuple(st.T)]]
    if not len(st):
        return None, "no reachable outside-mask start"
    mcp = MCP_Geometric(np.where(allowed, 1.0, np.inf), sampling=tuple(sp))
    gd, _ = mcp.find_costs([tuple(x) for x in st])
    gr = np.isfinite(gd) & (gd <= a.grow_mm) & allowed
    if gr.sum() < a.min_cand_voxels:
        return None, "too small after growth"
    reach = float(gd[gr].max())
    radial_reach = float(d[gr].max())
    if reach < a.min_reach_mm:
        return None, f"outside path reach {reach:.1f} < {a.min_reach_mm} mm"
    if radial_reach < a.min_radial_reach_mm:
        return None, f"radial reach {radial_reach:.1f} < {a.min_radial_reach_mm} mm"

    prox = gr & (gd <= max(a.min_reach_mm, a.seed_mm))
    p75 = float(np.percentile(v[prox], 75)) if prox.any() else -1000.0
    br = (p75 - c["soft_hu"]) / max(c["lumen_hu"] - c["soft_hu"], 1.0)

    patch = np.asarray(b["wall_patch"])
    if len(patch) > 1:
        xyz = patch * sp
        centre = xyz.mean(axis=0)
        r = np.linalg.norm(xyz - centre, axis=1)
        diam = float(2.0 * np.percentile(r, 75))
    else:
        diam = float(sp.min())
    diam = max(diam, float(sp.min()))
    if diam < a.min_ostium_mm:
        return None, f"ostium {diam:.1f} < {a.min_ostium_mm} mm"

    x = dict(b)
    x.update(
        grown=gr,
        gdist=gd,
        mcp=mcp,
        reach=reach,
        radial_reach=radial_reach,
        bright_ratio=br,
        ostium_diam_mm=diam,
    )
    return x, None


def signature(r, p, sp):
    z, y, x = [int(np.clip(round(q), 0, r.shape[k] - 1)) for k, q in enumerate(p)]
    spec = [
        (0, z, (y, x), sp[1] * sp[2]),
        (1, y, (z, x), sp[0] * sp[2]),
        (2, x, (z, y), sp[0] * sp[1]),
    ]
    out = []
    for ax, i, (u, w), area in spec:
        pl = r.take(i, axis=ax)
        if not pl[u, w]:
            return None, None
        lab, _ = ndimage.label(pl, STRUCT2)
        li = lab[u, w]
        if not li:
            return None, None
        n = (lab == li).sum()
        out.append((2 * np.sqrt(max(n, 1) * area / np.pi), ax))
    out.sort()
    return [q[0] for q in out], out[0][1]


def recenter(field, p, sp, mm):
    p = np.rint(p).astype(int)
    w = np.maximum(np.ceil(mm / sp).astype(int), 1)
    sl = tuple(
        slice(max(p[k] - w[k], 0), min(p[k] + w[k] + 1, field.shape[k]))
        for k in range(3)
    )
    sub = field[sl]
    zz, yy, xx = np.mgrid[sl[0], sl[1], sl[2]]
    dist = np.sqrt(
        ((zz - p[0]) * sp[0]) ** 2
        + ((yy - p[1]) * sp[1]) ** 2
        + ((xx - p[2]) * sp[2]) ** 2
    )
    cand = np.where(dist <= mm, sub, -1)
    o = np.unravel_index(int(np.argmax(cand)), cand.shape)
    return np.array([sl[k].start + o[k] for k in range(3)], float)


def bifurcation(c, a, sp):
    gd, gr = c["gdist"], c["grown"]
    step = max(a.bifurcation_step_mm, sp.min())
    half = max(0.75 * sp.min(), 0.6 * step)
    stable = 0
    first = None
    for dist_mm in np.arange(max(2.0, 2 * step), a.trace_mm + 1e-6, step):
        lab, n = ndimage.label(gr & (np.abs(gd - dist_mm) <= half), STRUCT3)
        sizes = ndimage.sum(lab > 0, lab, index=np.arange(1, n + 1)) if n else []
        branches = sum(np.asarray(sizes) >= 2)
        if branches >= 2:
            stable += 1
            first = dist_mm if first is None else first
            if stable >= 2:
                return float(first)
        else:
            stable = 0
            first = None
    return None


def branch_angle_deg(c, ostium_idx, seed_idx):
    sp = c["spacing"]
    z = int(np.clip(round(float(ostium_idx[0])), 0, c["mask"].shape[0] - 1))
    aorta_axis = unit(c["aorta_tangent_zyx"][z])
    daughter_axis = unit((np.asarray(seed_idx) - np.asarray(ostium_idx)) * sp)
    dot = float(np.clip(abs(np.dot(aorta_axis, daughter_axis)), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def outward_progress_fraction(c, path):
    if len(path) < 3:
        return 0.0
    idx = np.rint(path).astype(int)
    for k in range(3):
        idx[:, k] = np.clip(idx[:, k], 0, c["dout"].shape[k] - 1)
    vals = c["dout"][tuple(idx.T)]
    if len(vals) >= 3:
        vals = ndimage.median_filter(vals, size=3, mode="nearest")
    dif = np.diff(vals)
    tol = 0.25 * float(c["spacing"].min())
    return float(np.mean(dif >= -tol)) if len(dif) else 0.0


def measure(c, b, a):
    sp = c["spacing"]
    gd, gr = b["gdist"], b["grown"]
    bf = bifurcation(b, a, sp)
    b["bifurcation_mm"] = bf
    lim = min(a.trace_mm, bf) if bf is not None and bf > a.seed_mm else a.trace_mm
    win = gr & (gd <= lim)
    if not win.any():
        return None, "empty trace"

    pts = np.argwhere(win)
    radial = c["dout"][win]
    best_rad = radial.max()
    far_pool = pts[radial >= best_rad - max(0.5, sp.min())]
    if not len(far_pool):
        return None, "no outward trace point"
    far = far_pool[int(np.argmax(gd[tuple(far_pool.T)]))]

    try:
        path = np.asarray(b["mcp"].traceback(tuple(far)), float)
    except Exception:
        return None, "traceback failed"
    if len(path) < 2:
        return None, "path too short"

    path = path[::-1].copy()
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0) * sp, axis=1))]
    if arc[-1] < a.seed_mm:
        return None, f"path {arc[-1]:.1f} < seed distance {a.seed_mm} mm"

    progress = outward_progress_fraction(c, path)
    if progress < a.min_outward_progress_frac:
        return None, f"outward progress {progress:.2f} < {a.min_outward_progress_frac:.2f}"

    seed = path[int(np.argmin(np.abs(arc - a.seed_mm)))]
    seed_i = np.rint(seed).astype(int)
    for k in range(3):
        seed_i[k] = np.clip(seed_i[k], 0, c["dout"].shape[k] - 1)
    if c["dout"][tuple(seed_i)] < min(a.min_radial_reach_mm, a.seed_mm) * 0.65:
        return None, "5 mm seed stayed too close to aorta"

    db = ndimage.distance_transform_edt(gr, sampling=sp)
    sg, _ = signature(gr, seed, sp)
    seed = recenter(db, seed, sp, float(np.clip(0.6 * sg[0], 1, 4)) if sg else 1.0)
    dims, _axis = signature(gr, seed, sp)
    if dims is None:
        return None, "signature undefined"

    d0, d1, d2 = dims
    anis = d2 / max(d0, 1e-6)
    elong = b["reach"] / max(d0, 1e-6)
    angle = branch_angle_deg(c, b["ostium_idx"], seed)
    daughter_dir = unit((seed - np.asarray(b["ostium_idx"])) * sp)
    outward_dot = float(np.dot(local_wall_normal(c, b["ostium_idx"]), daughter_dir))

    b.update(
        d_min=d0,
        d_mid=d1,
        d_max=d2,
        anisotropy=anis,
        elongation=elong,
        branch_angle_deg=angle,
        outward_progress_frac=progress,
        outward_dot=outward_dot,
    )

    if d0 < a.min_caliber_mm:
        return None, "caliber too small"
    if d0 > a.max_caliber_mm:
        return None, "caliber too large"
    if anis < a.min_anisotropy:
        return None, "blob-like shape"
    if elong < a.min_elongation:
        return None, "not elongated"
    if b["bright_ratio"] < a.min_bright_ratio:
        return None, "too dim"
    if outward_dot < a.min_outward_dot:
        return None, f"outward dot {outward_dot:.2f} < {a.min_outward_dot:.2f}"
    if angle < a.min_branch_angle_deg:
        return None, f"branch angle {angle:.1f} < {a.min_branch_angle_deg:.1f} deg"

    zi, yi, xi = [int(np.clip(round(q), 0, gr.shape[k] - 1)) for k, q in enumerate(seed)]
    ost = to_phys(c, b["ostium_idx"])
    sm = to_phys(c, seed)
    direction = unit(sm - ost)
    b.update(
        seed_idx=seed,
        ostium_mm=ost,
        seed_mm=sm,
        radius_mm=max(float(db[zi, yi, xi]), a.min_radius_mm),
        direction=direction,
        path=path,
    )
    return b, None


def deduplicate(found, a):
    """Only collapse near-identical openings; do not merge distinct nearby origins."""
    kept = []
    for b in sorted(found, key=lambda x: (-x["radial_reach"], -x["bright_ratio"])):
        duplicate = False
        for q in kept:
            d = np.linalg.norm(b["ostium_mm"] - q["ostium_mm"])
            if d <= a.dedup_mm:
                align = float(np.dot(b["direction"], q["direction"]))
                if align > 0.90:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(b)
    return kept


def emit(caseid, found, path):
    found = sorted(found, key=lambda x: -x["ostium_mm"][2])
    ds = []
    for i, b in enumerate(found, 1):
        ds.append(
            dict(
                instance_id=f"branch_{i:03d}",
                parent_instance_id="aorta",
                ostium_xyz_mm=[round(float(x), 3) for x in b["ostium_mm"]],
                seed_xyz_mm=[round(float(x), 3) for x in b["seed_mm"]],
                radius_mm=round(float(b["radius_mm"]), 3),
                direction_xyz=[round(float(x), 5) for x in b["direction"]],
            )
        )
    out = {"case_id": caseid, "parent": {"instance_id": "aorta"}, "daughters": ds}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def viz(c, found, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v, m = c["vol"], c["mask"]
    views = [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (axis, (q0, q1)) in zip(axs, views):
        ax.imshow(v.max(axis), cmap="gray", vmin=c["soft_hu"] - 120, vmax=c["lumen_hu"] + 120)
        ax.contour(m.max(axis).astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        for b in found:
            oy, ox = b["ostium_idx"][q0], b["ostium_idx"][q1]
            sy, sx = b["seed_idx"][q0], b["seed_idx"][q1]
            ax.plot(ox, oy, "ro", mfc="none")
            ax.annotate("", xy=(sx, sy), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color="r"))
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main(argv=None):
    a = args_parse(argv)
    t = time.time()
    c = geometry(intensity(load_case(a.image, a.mask, a.margin_mm, a.debug), a), a)
    bases = candidate_openings(c, a)
    grown = []
    rejects = []
    for x in bases:
        y, why = grow(c, x, a)
        if y is None:
            rejects.append(why)
        else:
            grown.append(y)

    found = []
    for x in grown:
        y, why = measure(c, x, a)
        if y is None:
            rejects.append(why)
        else:
            found.append(y)

    before_dedup = len(found)
    found = deduplicate(found, a)
    if before_dedup > len(found):
        rejects.extend(["near-identical duplicate"] * (before_dedup - len(found)))

    caseid = os.path.basename(os.path.dirname(os.path.abspath(a.image))) or "case"
    out = emit(caseid, found, a.output)
    if a.viz:
        viz(c, found, a.viz)
    if a.debug:
        log(True, Counter(rejects))
        for i, b in enumerate(found, 1):
            log(
                True,
                f"keep {i}: angle={b['branch_angle_deg']:.1f}deg "
                f"radial={b['radial_reach']:.1f}mm progress={b['outward_progress_frac']:.2f} "
                f"outdot={b['outward_dot']:.2f} diam={b['ostium_diam_mm']:.1f}mm",
            )
    print(
        f"{caseid}: {len(out['daughters'])} daughters "
        f"({len(bases)} outward-tube candidates, {len(rejects)} rejected) "
        f"in {time.time() - t:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
