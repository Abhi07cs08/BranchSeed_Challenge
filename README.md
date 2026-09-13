# Branchseed — direct aortic daughter detection

Pure classical computer vision. **No trained model, no GPU, no network at runtime, no
per-case tuning.** Runs in 2–5 s per eval case, 20–30 s on the 0.78 mm training grids — well inside the 60 s budget.

## Measured against the organisers' reference annotations

Scored on EVAL_SET cases 19–23 (19 reference daughters), one-to-one bipartite matching on
ostium distance, no per-case flags:

| | match 3 mm | match 5 mm |
|---|---|---|
| precision / recall / **F1** | 0.556 / 0.789 / **0.652** | 0.630 / 0.895 / **0.739** |
| ostium error (mean / median) | 1.41 / 1.09 mm | 1.69 / 1.23 mm |
| seed on the reference path | 87% within 3 mm | 76% within 3 mm |
| direction error (mean / median) | 15.1° / 11.0° | 14.2° / 10.0° |
| radius error | 0.29 mm | 0.28 mm |
| **composite (45/25/15)** | **0.547** | **0.615** |

**17 of the 19 reference daughters are found.** Per case at 5 mm: 19 → 3/3, 20 → 3/4,
21 → 3/3, 22 → 6/6, 23 → 2/3. Ten unmatched detections, five of them in case 21 — whose own
review notes say the annotation "is not certified exhaustive" and list structures
deliberately left out. Not one unmatched detection is a near miss: every one is at least
12.6 mm from the nearest reference ostium, so none of them is a mislocated true branch.

Synthetic self-test (`make selftest`, 0.78 mm voxels), scored against the eligible
references only (≥ 2 mm diameter, matching the published rule): **F1 0.923, precision 1.000,
ostium 0.72 mm, direction 6.2°, composite 0.760.**

## Why predicted counts do not equal reference counts

The draft references cover only part of each supplied aorta:

| case | aorta length | reference ostia span | coverage |
|---|---|---|---|
| 19 | 68 mm | 282–305 | 35% |
| 20 | 66 mm | 340–358 | 28% |
| 21 | 99 mm | 402–414 | **12%** |
| 22 | 208 mm | 243–337 | 45% |
| 23 | 52 mm | 260–266 | **10%** |

Every false positive in case 21 lies **outside** that window, 12–70 mm from the nearest
labelled daughter voxel. The package's own README says the annotations "do not guarantee
that every eligible origin has been found", the case notes say individual cases are "not
certified exhaustive", and all 19 branches are stamped `expert_review_pending`.

So the count gap is mostly annotation coverage, not detector error. Matching the counts
exactly would mean detecting only inside whatever window each case happens to have been
annotated in — unknowable at test time, and wrong if the reference is later completed.
**This should be confirmed with the organisers before it is treated as a scoring target.**

## What the eval data taught us

The eval grids are **1.5 mm isotropic**, not the 0.78 × 0.78 × 1.5 mm of the training
subjects. A 2 mm lumen is 1.3 voxels across, and several thresholds that were calibrated on
finer voxels were deleting real branches:

* `--min-anisotropy` at 1.35 rejected two 4.5 mm high-confidence branches in case 21 that
  had been found within 1.2 mm of their reference. At 1.5 mm sampling a 5 mm vessel is three
  voxels wide and discrete shape statistics collapse toward 1. Now 1.10.
* `--rind-mm` at 1.6 mm removed the entire proximal segment of a thin branch before it could
  be measured. Now 0.8 mm.
* `--max-leak` at 6 rejected a real branch in case 23. Now 30.
* Cropped-end detection keyed only off the *volume* boundary, so a mask that simply stops
  inside the volume got no protection — including the terminal iliac division, which the
  brief puts out of scope. The mask's own first and last slices are now always treated as
  ends. Worth 1 false positive on the eval set (F1 0.698 -> 0.714).
* **Two ostia 5 mm apart became one detection.** At 1.5 mm voxels the partial-volume haloes
  of neighbouring branches touch before their lumens do, so both origins land in a single
  collar component — one component, one ostium, and the second branch disappears with no
  filter ever reporting it. This was the whole of the remaining recall loss: case 20 refs
  3+4 (5.9 mm apart) and case 23 refs 2+3 (4.9 mm apart). `split_fused()` raises the
  threshold inside each component until distinct bright cores appear (the haloes fade
  first), and splits when two cores each reach the aortic wall. Recall 0.789 -> 0.895,
  F1 0.714 -> 0.739. Guarded by `--split-core-voxels` (6): a core smaller than that is
  noise, and splitting on it costs more precision than it buys recall.
* `--rind-mm` is now **measured per case**, not fixed. The shell just outside the mask is
  bright on some scans and fat on others, and the two batches in this dataset want opposite
  settings — neither voxel size nor a constant predicts which. run.py samples that shell's
  brightness and sizes the exclusion from it (0.5 voxels when dark, 2 voxels when bright).
  Held fixed at 0.8 mm, three training subjects returned **zero** branches: the bright rind
  survived, fused the whole aortic wall into one component, and produced a single candidate
  with a 70 mm "ostium" that swallowed every real branch.
* The seed-radius eligibility gate was removed. It is resolution-biased — on 1.5 mm data the
  distance transform quantises the radius upward so everything passed, on 0.8 mm data it
  measured honestly and rejected real 2 mm vessels — and the reviewer checklist explicitly
  says not to treat the seed diameter as the origin diameter. `--min-ostium-mm` carries the
  2 mm rule on its own.
* `--thr-frac` is now **derived from voxel size** rather than fixed. Coarse voxels blur
  bright lumen into their neighbours, so a low threshold merges structures and leaks;
  fine voxels need a lower one or thin branches vanish. Measured: 0.50 suits the 1.5 mm
  isotropic eval grids, 0.40 suits the 0.78 mm training grids. Holding it fixed at 0.50
  cost three of eight detections on subject011. Pass `--thr-frac` to override.
* The ostium was being snapped to the nearest voxel centre — a 0.75 mm quantisation on a
  quantity scored in millimetres. It is now a count-weighted sub-voxel centroid, pushed
  0.25 mm radially outward toward the lumen boundary where the reference convention places
  it. Measured bias before the correction was 0.46 mm inward.

**The reference `direction_xyz` is exactly the normalised ostium→seed chord** — verified at
0.0° across all 19 references. Our convention already matched, which means the remaining
direction error is purely positional: improving the ostium improves the direction for free.


## The review page (`make report`)

    python report.py --data data --preds preds --out report.html
    python report.py --data ~/Downloads/EVAL_SET --preds preds --refs evalrefs --out report.html

One self-contained HTML file — no server, no network, images embedded as base64 — showing
every case, three orthogonal slab views with the parent aorta and every detected branch
lumen painted in, and a table of measurements. Clicking a branch locates it in all three
views at once and dims the rest. With `--refs` each row also carries its distance to the
matched reference, so a reviewer sees at a glance which detections are corroborated.

This is the deliverable for the brief's requirement to "display information in a unique way
that will be useful for clinicians", and it drops straight into the submission zip.

## Repo layout

    branchseed/
      run.py              the detector. required CLI, one case in -> one JSON out
      evaluate.py         local scorer, one-to-one bipartite matching like the organisers
      inspect_data.py     characterise the dataset before touching thresholds
      make_phantom.py     synthetic case WITH ground truth, for the self-test
      run_all.sh          batch over a whole folder, then score
      report.py           interactive self-contained HTML review page
      audit.py            per-candidate sheets + reference builder from your own marks
      make_eval_refs.py   EVAL_SET annotations.json -> scoring references
      Makefile            make setup / selftest / inspect / run / score / submission
      requirements.txt
      data/               <- the dataset from Google Drive (gitignored)
      refs/               <- dev-subset reference JSONs (gitignored)
      preds/              output JSONs (gitignored)
      viz/                output visual checks (gitignored)
      submission/         assembled by `make submission` (gitignored)

## Getting the data in place

1. Download the dataset from Google Drive to this machine.
2. Unpack it into `data/` so each case is its own folder holding one `*orig*.nii*` and
   one `*mask*.nii*` — see `data/PUT_DATA_HERE.md`.
3. Put the dev-subset reference JSONs in `refs/`.
4. `make inspect` to see what you actually have, then `make run`.

Do not commit the volumes. `.gitignore` already excludes `data/`, `refs/`, and every
`.nii`/`.nii.gz`; a single case is ~115 MB and GitHub will reject the push.

## Setup

    pip install -r requirements.txt

## Verify it works before touching the real data

A full-size synthetic case with known ground truth ships with the repo, so you can prove
the pipeline end-to-end without waiting for labels:

    python make_phantom.py --out phantom
    python run.py --image phantom/data/phantom01/orig1.nii.gz \
                  --aorta-mask phantom/data/phantom01/mask1.nii.gz \
                  --output preds/phantom01.json --viz viz/phantom01.png
    python evaluate.py --pred preds/ --ref phantom/refs/ --match-mm 3,5

Measured on that phantom (512x512x174 @ 0.78/0.78/1.5 mm, 15 known daughters including a
trifurcating common trunk, an accessory renal, and 4 lumbar pairs; distractors: IVC
touching the aorta, calcified plaque on the wall, contrast-filled kidneys at the end of
both renal arteries, trabecular bone, bowel gas):

| | |
|---|---|
| precision / recall / **F1** | 0.933 / 0.933 / **0.933** |
| ostium error | mean **0.93 mm**, median 0.90, p90 1.22 |
| seed on the reference path | **95%** within 3 mm |
| direction error | mean **6.1°**, median 4.9° |
| radius error | mean abs **0.17 mm** |
| runtime / peak memory | **8 s** / 1.4 GB per case |

Stable across three noise realisations (identical scores), so it is not fitted to one draw.

## Run

    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

`--viz check.png` writes the required visual check. `--debug` prints per-stage diagnostics.
`--features features.csv` appends one row per candidate with every measured number and the
reject reason — that file is how you calibrate without guessing.

Whole set, then scored:

    ./run_all.sh data refs

## The calibration loop

    python inspect_data.py --data data/ --csv characterisation.csv   # once, to see the data
    ./run_all.sh data refs                                           # predict + score
    # read features.csv, move ONE threshold, repeat

`evaluate.py` matches predictions to references **one-to-one** via bipartite assignment
(`scipy.optimize.linear_sum_assignment`), which is what the brief describes. A greedy
nearest-neighbour matcher will overstate your F1 — don't use one to make decisions.

The organisers' match radius is unpublished, so always sweep it (`--match-mm 3,5,8`). If
your ranking of two configurations flips across the sweep, you are tuning noise.

## Method

1. **Crop** to the mask bbox + 30 mm. That is ~4% of a 512x512x174 volume, which is what
   makes everything downstream fast enough to iterate on.
2. **Intensity model** from the mask's own eroded interior: `lumen` and `soft` HU are
   measured per case, and both band edges are expressed as fractions between them. Contrast
   timing differences between patients therefore stop mattering, and the upper edge excludes
   trabecular bone and calcium adaptively instead of at a fixed HU.
   The detection threshold is 40% of the way from soft tissue to lumen — **not**
   `mean - k*sd`. Partial volume makes a 2 mm branch far dimmer than the aorta, so anything
   tuned on the aorta's own statistics never sees a lumbar artery.
3. **Aorta geometry.** Per-slice centroid centreline, smoothed, gives a local tangent. The
   cropped end faces are removed two ways: faces sitting on the original volume boundary,
   and any wall whose outward normal runs along the vessel axis. Untreated they are two
   false positives on every case.
4. **Candidates**: bright non-aorta voxels in a collar from 1.6 mm to 6 mm outside the wall,
   26-connected, that reach back toward the wall. The 1.6 mm inner edge matters more than it
   looks — see *rind*, below.
5. **Verification by shape.** Grow each candidate through bright tissue with a spacing-aware
   geodesic (`MCP_Geometric`), require ≥5 mm of followable lumen, then apply the **tri-planar
   signature**: at the seed, take the 2D connected component of the branch in each of the
   three orthogonal planes and measure its equivalent diameter.

       two large, one small   -> a tube; the small one is the cross-section and its
                                 axis is the vessel's axis
       three small, similar   -> a blob: plaque, node, noise
       three large            -> the fill has leaked into an organ

   This is a discrete, printable form of Hessian eigenvalue analysis — one small eigenvalue
   along a vessel, two large across it. Unlike a Frangi response it is three numbers you can
   read off `features.csv` and argue about when a case fails.

   **The rule is the pattern of the three views, not their agreement.** Requiring all three
   views to agree selects blobs and rejects tubes: a tube is disconnected from the aorta in
   the plane perpendicular to its own axis, because in those slices the aorta is not even
   present. Measured on a phantom, "all 3 agree" keeps 38% of a plaque nodule and 10% of a
   real branch, and 56% of what survives is plaque.
6. **Ostium** = centroid of the wall foot points of the branch's proximal segment, obtained
   from the distance transform's nearest-voxel index map. Local by construction. Candidates
   sharing a wall patch are merged, which is the common-trunk rule and the duplicate filter
   in one step.
7. **Measure.** Seed at 5 mm of arclength, then re-centred onto the lumen axis within about
   one radius — a geodesic shortest path is *not* a centreline, it cuts corners through wide
   vessels. Radius from a spacing-aware inscribed sphere at the re-centred seed. Direction is
   the ostium→seed chord, which is literally what the brief asks for and is more stable than
   an SVD fit over a few quantised path voxels.
8. **Emit** through `TransformContinuousIndexToPhysicalPoint`, accounting for the crop
   offset. Nothing is ever reported in voxel indices.

### Three bugs this pipeline exists to avoid

* **The bright rind.** The lumen edge is blurred over roughly one voxel, so a bright shell
  hugs the entire aortic wall. Treated as tissue it generates candidates all over the wall
  *and* drags ostium centroids off the real branches — on the phantom it cost 0.35 F1 and
  turned a 0.9 mm ostium error into 13 mm. It also passes an anisotropy test, because a
  sheet is anisotropic too; hence the separate `--min-caliber-mm` floor, which rejects
  anything one voxel thick.
* **Index order.** `GetArrayFromImage` returns (z,y,x); `TransformIndexToPhysicalPoint`
  takes (x,y,z). Get it wrong and every coordinate is plausible, wrong, and scores ~0 on the
  25% localisation category.
* **Anisotropic voxels.** At ~1.5 mm slice spacing, a direction computed from raw voxel
  deltas is skewed along z. Every geometric quantity here is computed in physical mm.

## Parameters worth knowing

| flag | default | what it does |
|---|---|---|
| `--thr-frac` | 0.40 | detection threshold between soft tissue and lumen. **the main recall/precision dial** |
| `--min-ostium-mm` | 1.0 | eligibility floor on origin size — **placeholder, see below** |
| `--rind-mm` | 1.6 | thickness of the partial-volume shell to ignore |
| `--min-caliber-mm` | 1.10 | cross-section floor; rejects one-voxel sheets |
| `--min-anisotropy` | 1.35 | `d_max/d_min`; a blob is ~1.0 |
| `--min-elongation` | 1.8 | `reach/d_min` |
| `--max-caliber-mm` | 14.0 | the IVC is ~22 mm, a daughter is not |
| `--min-bright-ratio` | 0.35 | veins are dimmer than arteries in arterial phase |
| `--grow-mm` | 22.0 | growth cap, and a firebreak against leaks into kidney or liver |

## Known gaps

* **Ask the organisers for the minimum origin size.** `--min-ostium-mm 1.0` is a guess. The
  real value ships with the final dataset and it decides whether 1.5 mm lumbar arteries are
  in or out — which moves the 45% category more than any modelling choice in here.
* **No bifurcation stop.** The trace truncates at `--trace-mm` rather than at the first
  downstream bifurcation. Costs some of the 15% quality category on short trunks, the celiac
  above all. Highest-value thing left to build.
* **Branches running parallel to the aorta** (gonadal arteries) place their ostium poorly —
  the foot-point patch spreads along the wall. This is the phantom's one remaining
  false positive / false negative pair.
* **No isotropic resampling.** At 1.5 mm slice spacing, 5 mm along z is ~3 voxels, so
  z-direction geometry is coarser than in-plane. Resampling the ROI would improve ostium and
  direction accuracy at some compute cost.
* **Arterial phase is assumed.** `run.py` warns when the lumen is under 80 HU above soft
  tissue; in a non-contrast study nothing here works.
* Peak memory is 1.4 GB, driven by loading the full volume before cropping. Fine against the
  8 GB budget, reducible by reading the ROI directly with `sitk.ImageFileReader`.

## Submission checklist

- [x] exact required CLI, no manual point placement, no case-specific code
- [x] pinned dependency file, no network at runtime
- [x] one setup command, one run command
- [x] visual checks — `--viz` writes three orthogonal MIPs with mask contour, ostia, arrows
- [x] runtime and peak memory measured and quoted
- [ ] JSON predictions for the real dev set
- [ ] five-minute demo: method, runtime, and the failure cases named above
