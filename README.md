# Drone LiDAR Seedling Analysis — Complete Project Reference

A pipeline for locating, measuring, and segmenting 444 individually-surveyed
seedlings from drone LiDAR flown at two altitudes (30m and 60m), now in a
**LAS-only** form (no RGB photos used for processing at all — see "Two
pipeline versions" below for why this exists alongside an earlier version).

This document reflects the *current* state of the whole project, including
two real production crashes that were found and fixed during development —
described honestly below, not glossed over, because understanding what broke
and why is part of trusting what's shipped now.

---

## Who this is for

Written for a newcomer to follow start-to-finish, and as a fast lookup for
anyone who already knows the domain and just needs one exact number or
config value. If a term is unfamiliar, check the **Glossary** near the end.

---

## The data

| What | Details |
|---|---|
| **Study area** | ~2,700 m² dense young conifer plantation, planted in rows |
| **Species** | All 444 surveyed trees: *Küstentanne* (grand fir) |
| **Tree spacing** | Median nearest-neighbor distance 1.53 m (10th pct 1.05m) |
| **CRS** | UTM Zone 32N (EPSG:32632) — consistent across every file |
| **Ground truth** | `trees_crop.shp` — each tree's surveyed base position. `PT_HEIGHT` is ground elevation at the base, **not** tree height (confirmed: matches the point geometry's Z exactly) |
| **LiDAR** | 2 flights at 60m, 2 at 30m, same area, ~20-50 min apart. Point format 7 (RGB + classification + multi-return + intensity + GPS time). Ground already classified (2=ground, 1=other) |
| **Coordinate precision floor** | Confirmed via the LAS header's scale factor: **1cm** — nothing in this data can be stored more precisely than that, regardless of point density |
| **Real point density** (median nearest-neighbor spacing, not a density approximation) | 60m: 2.00cm. 30m: 1.00cm (sits right at the file's own precision floor) |
| **RGB coverage within LAS points** | Only **71.2%** of vegetation-classified points have real (non-zero) embedded color — a genuine, confirmed limitation, not a processing artifact |
| **Drone photos** | 1 photo at 60m, 5 at 30m, 1cm/pixel. Each independently "roughly" georeferenced (your own description, confirmed by testing) — **not used by the processing pipeline**, only by the optional verification tool (script 11) |

### Required folder layout

```
your_project_folder/
├── trees_crop.shp / .shx / .dbf / .prj
├── 60m/
│   ├── 260522_122413_filt_crop.las
│   ├── 260522_124721_filt_crop.las
│   └── DSC00411_georef_1cm_crop.tif        (only used by script 11)
├── 30m/
│   ├── 260522_130956_filt_crop.las
│   ├── 260522_140126_filt_crop.las
│   └── DSC02145 through DSC02149_georef_1cm_crop.tif   (only used by script 11)
├── 06b_stage1_buffer_localize_LAS_ONLY.py
├── 07b_stage2_height_extraction_LAS_ONLY.py
├── 08b_stage3_crown_segmentation_LAS_ONLY.py
├── 09_las_attribute_analysis.py
├── 10_stage1_fallback_diagnostic.py
└── 11_verify_against_drone_tif.py
```

### Installing dependencies

```bash
pip install laspy numpy scipy rasterio matplotlib pyshp shapely scikit-image pandas
```

---

## Two pipeline versions — which one to actually run

| | **LAS-only (06b/07b/08b)** — current, recommended | Original (06/07/08) — superseded |
|---|---|---|
| RGB source | The LAS points' own embedded color | The 6 drone `.tif` photos |
| Status | Actively developed, most recent fixes | Kept for reference/comparison only |
| Use this if | You want a pipeline that works even with no companion photos at all, or you specifically want LAS-only processing | You want to compare against the earlier photo-based results |

**Run the LAS-only version unless you have a specific reason not to.** The
rest of this document covers it in full; the original TIF-based scripts
still exist and still work, but aren't being actively extended.

---

## The pipeline, in order

```
09 (optional, informational)
06b → 07b → 08b
        ↑        ↑
   10 (diagnostic)   11 (diagnostic)
```

**06b → 07b → 08b must run in that order** — each reads the previous
stage's output CSV. **09, 10, and 11 are independent diagnostic tools**:
09 can be run any time (just needs the LAS files), 10 and 11 both need
06b's output to already exist, but don't feed into 07b/08b at all — they're
for your own visual verification, not part of the processing chain.

---

## Script 09 — `09_las_attribute_analysis.py`: what's actually in your LAS files

Run this first, or any time you want a refresher. Three things, all saved
to `outputs/las_analysis/` (not just printed — confirmed and fixed early
on, since the first version silently produced nothing on disk):

1. **Every attribute listed and described**, cross-checked against what's
   *actually populated* vs. just technically present. Confirmed on real
   data: 8 fields (`synthetic`, `key_point`, `withheld`, `overlap`,
   `scanner_channel`, `scan_direction_flag`, `user_data`, and
   `point_source_id` within any single file) are always `0` — present per
   the LAS spec, never used by this data's provider.
2. **A usage recommendation table** — ESSENTIAL / WORTH TESTING / NOT
   USEFUL per attribute, grounded in what's really populated.
3. **Real point density and a resolution suggestion per condition** — this
   is where the per-condition resolution table used throughout the rest of
   the pipeline actually comes from (see next section).

### Two specific, tested findings worth knowing before you look at anything else

- **RGB is effectively 8-bit precision**, not the full 16-bit the field
  allows (~256 unique values per channel, not 65,536) — same precision as
  an ordinary camera, confirmed directly.
- **Intensity and multi-return were tested as a way to distinguish a
  "log-like" (whitish) false positive from real green vegetation** — the
  direct test found intensity medians only ~1.4% apart between the two
  groups, and multi-return fraction near-zero for both. **Neither is used
  to drive any scoring in this pipeline** — they're recorded as diagnostic
  columns only, since the direct evidence didn't support using them as a
  primary signal (caveat: this specific test compared RGB-defined groups,
  so it can't rule out intensity helping when RGB itself is fooled — that
  needs an actual confirmed log location to test properly).

### Where the per-condition resolution table comes from

Real point-density testing established **three separate facts**, not one:
1. The **hard floor**: 1cm, from the LAS coordinate scale factor itself —
   meaningless to go finer than this regardless of anything else.
2. **Median nearest-neighbor spacing** (true, not approximated): 60m=2.00cm,
   30m=1.00cm.
3. **Points-per-cell at candidate resolutions** — tested directly to find
   where median color aggregation (used in Stage 1) actually has 2-4 real
   points to work with, not just 1.

Resulting table, used identically across Stage 1, 2, and 3:

| Condition | Resolution |
|---|---|
| 60m_flight_1 / 60m_flight_2 | 6cm |
| 30m_flight_1 / 30m_flight_2 | 4cm |
| 60m_merged | 5cm |
| 30m_merged | 3cm |
| all_4_merged | 2cm |

---

## Script 06b — Stage 1: find the real tree position

### What changed from a plain "search a circle" approach

1. **Genuinely cylindrical, not just a circle with a nicer name.** The XY
   radius (`buffer_radius_m`, 0.2m — half the original TIF-based version's
   0.4m) is fixed, but the Z dimension has **no upper bound at all** —
   any point above ground, however tall, is included when building the
   raster. There's a lower bound only for *color*-contributing points
   (0.03m above ground). Height/CHM itself uses a looser -0.05m noise
   margin. Not a fixed height band — confirmed directly from the code, not
   assumed.

2. **Color comes only from the LAS points' own RGB.** Aggregated per grid
   cell as the **median** color of every point in that cell's full
   vertical column (not just the topmost point) — more robust to a single
   stray/outlier-colored point.

3. **The color signal is local-contrast, not raw greenness.** A plain
   "how green is this pixel" measure gets dominated by a big uniform grass
   patch (confirmed as a real failure mode during earlier development,
   before the LAS-only redesign) — this version subtracts a locally-
   blurred version of the greenness signal from itself, so a small plant
   brighter than its *own* immediate surroundings stands out regardless of
   what else is in the buffer.

4. **A hard height ceiling** (`max_plausible_height_m`, 2.0m) stops the
   algorithm from ever preferring a taller neighboring object just because
   it has a stronger signal — but this only excludes *implausibly tall*
   things (like a real mature tree), not merely "taller than the seedling
   but still short" (like tall grass) — a real, acknowledged limitation.

5. **Two real memory crashes, both fixed, both confirmed via `dmesg`, not
   assumed:**
   - Building `all_4_merged`'s ground TIN from ~12 million un-thinned
     ground points OOM-killed the process (3.94-3.95GB in a 3.9GB
     sandbox). Fixed by thinning ground points to the CHM's own resolution
     before triangulating — the surface is only ever evaluated at that
     resolution anyway, so nothing real is lost.
   - This fix was originally only applied to Stage 2/3 and had to be
     ported back into Stage 1 separately after the same risk was found
     there too — worth remembering if you ever copy logic between these
     scripts again.

### The startup sequence (added later, now standard)

Every run now prints, before any real processing:
- **Every LAS attribute present**, read fresh from your actual files.
- **An interactive resolution prompt** for the active condition — press
  Enter for the recommended value (from the table above), or type a
  custom number in cm (10cm is suggested as a quick, coarser alternative).
  This only prompts once, for the primary/active condition — the other 6
  comparison-only conditions silently use their table defaults.

### The 3-tier "honesty" visualization

Real color where it exists; failing that, the cell's own **raw Z**
(explicitly the raw LAS attribute, not the ground-normalized CHM value —
this means the tint can drift with the plot's real terrain slope, not just
individual plant bumps, which is expected given what's being shown);
failing that, either a borrowed neighbor's data or plain black, controlled
by `use_neighbor_borrow_fallback`. Confirmed on real `all_4_merged` data:
39.9-41.3% of cells (varies slightly by condition/run) fall into the
raw-Z tier, and only 11.6-27.5% (varies by condition — finer resolutions
have relatively more truly-empty cells) need the last-resort borrow.

**This never fully matches a real drone photo's clarity** — confirmed by
direct comparison and by testing two different enhancement attempts
(smoothing, height-relief shading), neither of which meaningfully improved
small-feature visibility. That's a real information-density ceiling from
sparse point-based color, not a rendering trick left undiscovered — see
script 11 below for what actually does help.

### Config you're most likely to touch

| Parameter | Effect |
|---|---|
| `buffer_radius_m` | Search window size (0.2m) |
| `active_condition` | Which of the 7 conditions actually drives localization |
| `use_neighbor_borrow_fallback` | `True`=no black in the saved image, `False`=black means genuinely zero data |
| `proximity_decay_fraction` | How strongly nearby beats far when scoring candidates |
| `review_score_percentile` | How large the `review_recommended` flagged subset is |

### Outputs (full detail in OUTPUTS_EXPLAINED.md)

`stage1_localized_trees_LAS_only.csv`, 3 point/buffer shapefiles, and **14
GeoTIFFs** (a CHM + a color raster for each of the 7 conditions).

### Real result from testing

Method breakdown: 414 localized, 8 no-height-bump, 22 zero-score (out of
444). RGB coverage 82.2%. Median offset 0.050m, max 0.200m. Confirmed
identical across two independent machines running the same script.

### How to verify

Load the point/buffer shapefiles and the `all_4_merged` color raster in
QGIS. Prioritize the `review_recommended=True` subset. For a clearer visual
check than the LAS-derived raster can give, use script 11 instead (below).

---

## Script 07b — Stage 2: height, across all 7 conditions

Builds a CHM seven separate ways and extracts height for all 444 trees
under each, to directly answer: is 60m sufficient, does 30m change the
answer, does merging help?

### Real, tested result

| Condition | Median height |
|---|---|
| 60m_flight_1 | 0.044-0.045m |
| 60m_flight_2 | 0.053-0.056m |
| 30m_flight_1 | 0.088-0.089m |
| 30m_flight_2 | 0.087-0.088m |
| 60m_merged | 0.075-0.076m |
| 30m_merged | 0.102-0.103m |
| **all_4_merged** | **0.112-0.113m** |

(Small run-to-run variation reflects Stage 1's own tiny nondeterminism from
its ground-thinning fix interacting with slightly different upstream
positions — not a bug, just noise at the millimeter scale.)

60m alone systematically **underestimates** height — confirmed both
numerically and via a scatter plot showing points consistently above the
1:1 line, not just noisier. 30m and merged agree closely. Merged achieves
the best completeness (fewest zero-point/low-confidence trees). A Wilcoxon
signed-rank test confirms merged-vs-60m is extremely significant
(p<10⁻³⁰), while merged-vs-30m is only marginally significant (p≈0.026) —
statistically real but practically negligible.

### What this script also produces, beyond height numbers

- **Real merged `.las` files** you can open directly in CloudCompare —
  added specifically to let you verify a reported "merge gap" concern
  yourself. That investigation confirmed the raw point-level merge is
  provably complete (a grid-cell coverage check found zero cells present
  in 30m-alone but missing from the merged set) — any visual difference
  you might see between conditions is a downstream ground-surface fitting
  effect, not missing data.
- **A CHM GeoTIFF per condition** (7 total), loadable in QGIS.
- **`mean_intensity`/`pct_multireturn`** per tree per condition, as
  diagnostics (see script 09's findings on why these aren't used for
  scoring).

### Real crash history for this script (relevant if you ever modify it)

Two separate OOM crashes were found and fixed while building `all_4_merged`
specifically: first from concatenating all 4 files' point arrays while
still holding the originals in memory (fixed with an incremental,
per-source accumulation approach), then from the ground TIN's Delaunay
triangulation itself being built from ~12 million un-thinned points (fixed
by thinning first). Both confirmed via `dmesg`, not inferred.

---

## Script 08b — Stage 3: crown shape and diameter

Grows a crown polygon from each tree's Stage 1 position via multi-seed
watershed on the merged CHM. **Reuses Stage 2's CHM directly if it already
exists** (checked first, saves several minutes) — falls back to building
it fresh (with the same memory-safe approach) if not.

### The honest finding this stage is built around

**Only 24.8-26.1% of trees (varies slightly by run) get a genuinely
resolvable crown shape.** This isn't a bug to keep chasing — confirmed via
two independent tests: even at 4× coarser resolution, the median connected
component size in the "tall enough to be a crown" area stayed at 1 pixel,
because over half the entire landscape (widespread low grass, not just
seedlings) clears the height threshold, fragmenting into either a few huge
grass blobs or masses of isolated single-pixel islands. Rather than fake a
diameter from 1-2 noise pixels, the script explicitly flags
`resolvable=False` for anything under `min_resolvable_pixels` (15).

A second, independent quality layer: even among "resolvable" crowns, a
shape check (eccentricity/circularity) catches ones that are long thin
slivers rather than real compact shapes — confirmed real result:
93-96% of resolvable crowns (varies by run) also pass this check.

### Real result

107-116 of 444 trees resolvable (24.1-26.1%), median diameter 0.166-0.208m
among those, 93-96% also passing the shape check.

---

## Diagnostic tools (not part of the required pipeline order)

### Script 10 — `10_stage1_fallback_diagnostic.py`

Finds every tree where Stage 1 fell back (no confident candidate found)
and rebuilds a visual crop for each, so you can judge whether that's a
genuinely ambiguous location or a real miss — without manually
cross-referencing tree IDs yourself. A real bug was caught building this:
the first version used a 5mm grid cell, finer than the real ~1-2cm point
spacing, producing mostly-black, unreadable images — fixed by matching
the cell size to real spacing and adding gap-fill.

### Script 11 — `11_verify_against_drone_tif.py`

Samples the **real drone photos** (not the LAS-reconstructed raster) at
Stage 1's localized coordinates — confirmed, with an objective
cross-correlation test (not just eyeballing), that the photo and LiDAR
coordinate systems align to within ~4cm in a representative region,
negligible next to the 0.2m search buffer. This is the clearest way to
visually check localization quality; the LAS-only raster alone was tested
twice for improvements (smoothing, height-relief shading) and neither
meaningfully helped small-feature visibility — this script is the better
tool for that specific job, not a fallback because the LAS-only path
failed.

---

## Honest summary of what this pipeline can and can't do

- **Localization**: works well for most trees; ~20% flagged for review,
  and dense-grass areas remain genuinely hard even for a human eye.
- **Height**: reliable, statistically backed. 60m alone is insufficient;
  merged data is the best available single answer.
- **Crown segmentation/diameter**: reliable for about a quarter of trees;
  the rest are honestly unmeasurable as a 2D shape at this point density,
  not a failure of the method.
- **Visualization**: the LAS-only raster is a real, working tool but has a
  genuine information-density ceiling for spotting tiny individual plants;
  the real drone photos (via script 11) are the better tool for that
  specific need, confirmed to align well with the trustworthy LiDAR
  coordinates.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError` | Check `CONFIG` paths match your folder exactly |
| Process dies with no Python traceback | Check `dmesg` for an OOM kill before assuming a code bug — this has happened twice before, both fixed, but worth checking first if you modify the scripts |
| Script 10/11 can't find the Stage 1 CSV | Run `06b` first — both diagnostics depend on its output |
| A `.zip` upload won't extract with `unzip` | Check with `file <filename>` — may actually be gzip/tar |
| Crown diameter blank/NaN for most trees | Expected — check `resolvable`; an honest flag, not missing data |

---

## Glossary

- **UTM32N / CRS**: flat, meters-based coordinate system; confirmed
  consistent across every file in this project.
- **Cylindrical buffer**: fixed XY radius, unbounded Z upward, near-zero
  floor downward — not a fixed-height 3D shape.
- **CHM**: height above local ground (not absolute elevation) per grid
  cell — what the algorithm scores against.
- **Raw Z**: the actual LAS point elevation, untouched — what the
  visualization's height-fallback tier uses instead of CHM, per explicit
  design choice.
- **Watershed segmentation**: floods a height surface from multiple seed
  points simultaneously; where two floods meet becomes a crown boundary.
- **Median nearest-neighbor spacing**: the real, empirically-measured
  typical distance between adjacent points — distinct from a
  density-derived *average* spacing, which can give a different number for
  the same data (both are reported and reconciled in this project).
- **The 3-tier visualization**: real color → own raw Z (colormapped) →
  borrowed neighbor or black, in that priority order, each tier only used
  when the previous one has no real data for that cell.