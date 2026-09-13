#!/usr/bin/env python3
"""
report.py — one self-contained HTML review page for a whole run.

The brief asks the prototype to "display information in a unique way that will be useful
for clinicians". Three static PNGs do not do that. This builds a single file that opens in
any browser with no server and no network: every case, three orthogonal views, every
detected branch as a clickable row that highlights its ostium and direction in all three
views at once, with the measurements beside it.

    python report.py --data data --preds preds --out report.html
    python report.py --data ~/Downloads/EVAL_SET --preds bestpreds --refs evalrefs \
                     --out report.html      # adds per-branch matched/missed status

Images are embedded as base64 PNGs, so the file can be emailed or dropped in a submission
zip and it still works.
"""

import argparse, base64, glob, io, json, os, re, sys
import numpy as np
import run as bs

VIEWS = [("Axial", 0, 1, 2), ("Coronal", 1, 0, 2), ("Sagittal", 2, 0, 1)]


def find_pair(d):
    imgs = [f for f in glob.glob(os.path.join(d, "*.nii*"))
            if re.search(r"orig|image|ct", os.path.basename(f), re.I)]
    msks = [f for f in glob.glob(os.path.join(d, "*.nii*"))
            if re.search(r"mask|aorta|seg", os.path.basename(f), re.I)
            and not re.search(r"daughter|combined", os.path.basename(f), re.I)]
    return (sorted(imgs)[0], sorted(msks)[0]) if imgs and msks else (None, None)


def slab_png(vol, mask, det, d_out, axis, soft, lumen, slab_mm):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = np.where(d_out <= slab_mm, vol, -1000.0).max(axis=axis)
    lo, hi = soft - 100, lumen * 1.05
    g = np.clip((show - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([g, g, g])
    mk = mask.max(axis=axis)
    rgb[mk] = 0.55 * rgb[mk] + 0.45 * np.array([0.25, 0.66, 0.77])
    dm = det.max(axis=axis)
    rgb[dm] = 0.45 * rgb[dm] + 0.55 * np.array([1.0, 0.58, 0.0])
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(rgb, 0, 1), format="png")
    return base64.b64encode(buf.getvalue()).decode()


def build_case(case_dir, pred_path, ref_path, slab_mm):
    img_p, msk_p = find_pair(case_dir)
    if not img_p:
        return None
    args = bs.parse_args(["--image", img_p, "--aorta-mask", msk_p, "--output", os.devnull])
    case = bs.load_and_crop(img_p, msk_p, args.margin_mm, False)
    case = bs.intensity_model(case, args.core_erode_mm, args.thr_frac, args.ceiling_frac,
                              args.hu_ceiling, args.hu_ceiling_max, args.lumen_pct,
                              args.thr_max, False)
    case = bs.aorta_geometry(case, args.cap_margin_mm, args.cap_cos, False)
    labels, lab = bs.find_candidates(case, args.collar_mm, args.rind_mm, args.touch_mm,
                                     args.min_cand_voxels, False)
    alive = []
    for li in labels:
        c, _ = bs.grow(case, lab, li, args)
        if c is None: continue
        c, _ = bs.place_ostium(case, c, args)
        if c is None: continue
        alive.append(c)
    alive = bs.merge_trunks(case, alive, args.merge_mm)
    final = []
    for c in alive:
        c2, _ = bs.measure(case, c, args)
        if c2 is not None: final.append(c2)
    final.sort(key=lambda c: -c["ostium_mm"][2])

    det = np.zeros_like(case["mask"])
    for c in final: det |= c["grown"]
    sp = case["spacing"]
    imgs = {}
    for name, axis, a0, a1 in VIEWS:
        imgs[name] = dict(png=slab_png(case["vol"], case["mask"], det, case["d_out"],
                                       axis, case["soft_hu"], case["lumen_hu"], slab_mm),
                          h=case["vol"].shape[a0], w=case["vol"].shape[a1],
                          aspect=float(sp[a0] / sp[a1]), mm=float(sp[a1]))

    pred = json.load(open(pred_path)) if pred_path and os.path.exists(pred_path) else None
    refs = json.load(open(ref_path))["daughters"] if ref_path and os.path.exists(ref_path) else []
    Rm = np.array([r["ostium_xyz_mm"] for r in refs]) if refs else None

    rows = []
    for n, c in enumerate(final, 1):
        pts = {}
        for name, axis, a0, a1 in VIEWS:
            pts[name] = dict(oy=float(c["ostium_idx"][a0]), ox=float(c["ostium_idx"][a1]),
                             sy=float(c["seed_idx"][a0]), sx=float(c["seed_idx"][a1]))
        status = ""
        if Rm is not None and len(Rm):
            d = float(np.linalg.norm(Rm - c["ostium_mm"], axis=1).min())
            status = f"{d:.1f} mm" if d <= 5 else f"unmatched ({d:.0f} mm)"
        rows.append(dict(id=f"branch_{n:03d}", pts=pts,
                         ost=[round(float(v), 1) for v in c["ostium_mm"]],
                         seed=[round(float(v), 1) for v in c["seed_mm"]],
                         r=round(float(c["radius_mm"]), 2),
                         dirv=[round(float(v), 2) for v in c["direction"]],
                         cal=round(float(c.get("d_min", 0)), 1),
                         leak=round(float(c.get("leak", 0)), 1),
                         status=status))
    return dict(case_id=os.path.basename(case_dir.rstrip("/")), images=imgs, rows=rows,
                n_ref=len(refs), band=[round(case["thr"]), round(case["hu_ceiling"])],
                lumen=round(case["lumen_hu"]),
                spacing=[round(float(s), 2) for s in reversed(sp)])


def render(cases, out):
    esc = json.dumps(cases)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Branchseed review</title>
<style>
:root{--bg:#0f1216;--panel:#171b21;--line:#262c35;--ink:#e8eaed;--dim:#98a1ad;
--acc:#ff9500;--aorta:#40a8c4;--ok:#39b87a;--bad:#e8575c;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;
align-items:baseline;gap:18px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12.5px}
nav{display:flex;gap:6px;padding:12px 22px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
nav button{background:var(--panel);color:var(--dim);border:1px solid var(--line);
border-radius:6px;padding:6px 13px;cursor:pointer;font:inherit;font-size:13px}
nav button.on{background:var(--acc);color:#1a1204;border-color:var(--acc);font-weight:600}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:18px;padding:18px 22px}
@media(max-width:1080px){.wrap{grid-template-columns:1fr}}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:720px){.views{grid-template-columns:1fr}}
.view{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px}
.view h3{margin:0 0 7px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
color:var(--dim);font-weight:600}
.view svg{width:100%;height:auto;display:block;border-radius:4px;background:#000}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;color:var(--dim);font-size:10.5px;letter-spacing:.08em;
text-transform:uppercase;padding:0 8px 7px 0;border-bottom:1px solid var(--line)}
td{padding:7px 8px 7px 0;border-bottom:1px solid var(--line);font-family:var(--mono);
font-size:12px}
tbody tr{cursor:pointer}
tbody tr:hover{background:#1e232b}
tbody tr.sel{background:#2a2013;outline:1px solid var(--acc)}
.pill{padding:1px 7px;border-radius:99px;font-size:10.5px;font-family:var(--mono)}
.pill.ok{background:rgba(57,184,122,.16);color:var(--ok)}
.pill.bad{background:rgba(232,87,92,.16);color:var(--bad)}
.meta{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;margin-bottom:14px;font-size:12.5px;color:var(--dim)}
.meta b{color:var(--ink);font-family:var(--mono)}
.legend{display:flex;gap:14px;font-size:11.5px;color:var(--dim);padding:2px 22px 16px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;
margin-right:5px}
.empty{color:var(--dim);padding:26px;text-align:center}
</style></head><body>
<header><h1>Branchseed — detected aortic daughters</h1>
<span class="sub" id="hdr"></span></header>
<nav id="nav"></nav>
<div class="legend"><span><i class="sw" style="background:#40a8c4"></i>parent aorta</span>
<span><i class="sw" style="background:#ff9500"></i>detected branch lumen</span>
<span><i class="sw" style="background:#e8575c"></i>ostium &amp; direction</span></div>
<div class="wrap"><div class="views" id="views"></div>
<div><div class="meta" id="meta"></div>
<table><thead><tr><th>Branch</th><th>Ostium (LPS mm)</th><th>r</th><th>Caliber</th>
<th>Ref</th></tr></thead><tbody id="tb"></tbody></table></div></div>
<script>
const CASES = __DATA__;
let ci = 0, sel = null;
const nav = document.getElementById('nav');
CASES.forEach((c,i)=>{const b=document.createElement('button');
  b.textContent=c.case_id+' ('+c.rows.length+')';
  b.onclick=()=>{ci=i;sel=null;draw();};nav.appendChild(b);});
function draw(){
  const c = CASES[ci];
  [...nav.children].forEach((b,i)=>b.classList.toggle('on',i===ci));
  document.getElementById('hdr').textContent =
    CASES.length+' case(s) · '+c.spacing.join(' × ')+' mm voxels · band '+
    c.band[0]+'–'+c.band[1]+' HU · lumen '+c.lumen+' HU';
  document.getElementById('meta').innerHTML =
    '<b>'+c.rows.length+'</b> daughters detected'+
    (c.n_ref?' · <b>'+c.n_ref+'</b> in the reference':'')+
    '<br>Click a row to locate that branch in all three views.';
  const V=document.getElementById('views'); V.innerHTML='';
  ['Axial','Coronal','Sagittal'].forEach(name=>{
    const im=c.images[name];
    const d=document.createElement('div'); d.className='view';
    const H=im.h*im.aspect, S=1.6/im.mm;   // markers sized in mm, so all three views agree
    let s='<h3>'+name+'</h3><svg viewBox="0 0 '+im.w+' '+H+
      '" preserveAspectRatio="xMidYMid meet">'+
      '<image href="data:image/png;base64,'+im.png+'" x="0" y="0" width="'+im.w+
      '" height="'+H+'" />';
    c.rows.forEach((r,k)=>{
      const p=r.pts[name], y=p.oy*im.aspect, sy=p.sy*im.aspect;
      const dx=(p.sx-p.ox)*2.0, dy=(sy-y)*2.0;
      const on = sel===k, w = (on?1.6:0.9)*S, op = (sel===null||on)?1:0.25;
      s+='<g opacity="'+op+'"><line x1="'+p.ox+'" y1="'+y+'" x2="'+(p.ox+dx)+'" y2="'+(y+dy)+
         '" stroke="#e8575c" stroke-width="'+w+'"/>'+
         '<circle cx="'+p.ox+'" cy="'+y+'" r="'+((on?2.4:1.7)*S)+'" fill="none" '+
         'stroke="#e8575c" stroke-width="'+w+'"/>'+(on?'<circle cx="'+p.ox+'" cy="'+y+
         '" r="'+(5*S)+'" fill="none" stroke="#ff9500" stroke-width="'+(0.9*S)+'"/>':'')+'</g>';
    });
    d.innerHTML=s+'</svg>'; V.appendChild(d);
  });
  const tb=document.getElementById('tb'); tb.innerHTML='';
  if(!c.rows.length){tb.innerHTML='<tr><td colspan="5" class="empty">No eligible daughters detected.</td></tr>';return;}
  c.rows.forEach((r,k)=>{
    const tr=document.createElement('tr'); tr.className = sel===k?'sel':'';
    const matched = r.status && r.status.indexOf('unmatched')<0;
    tr.innerHTML='<td>'+r.id.replace('branch_','')+'</td><td>'+r.ost.join(', ')+
      '</td><td>'+r.r.toFixed(2)+'</td><td>'+r.cal.toFixed(1)+'</td><td>'+
      (r.status?'<span class="pill '+(matched?'ok':'bad')+'">'+r.status+'</span>':'—')+'</td>';
    tr.onclick=()=>{sel = sel===k?null:k; draw();};
    tb.appendChild(tr);
  });
}
draw();
</script></body></html>"""
    open(out, "w").write(html.replace("__DATA__", esc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--slab-mm", type=float, default=25.0)
    a = ap.parse_args()
    cases = []
    for d in sorted(glob.glob(os.path.join(os.path.expanduser(a.data), "*"))):
        if not os.path.isdir(d): continue
        cid = os.path.basename(d)
        pp = os.path.join(a.preds, f"{cid}.json") if a.preds else None
        rp = os.path.join(a.refs, f"{cid}.json") if a.refs else None
        c = build_case(d, pp, rp, a.slab_mm)
        if c: cases.append(c); print(f"  {cid}: {len(c['rows'])} daughters")
    if not cases: sys.exit("no cases found")
    render(cases, a.out)
    mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}  ({mb:.1f} MB, {len(cases)} cases) — open it in any browser")


if __name__ == "__main__":
    main()
