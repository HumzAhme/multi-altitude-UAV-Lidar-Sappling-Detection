# What Every Generated File and Column Means (LAS-only pipeline)

Complete reference for scripts `06b`, `07b`, `08b`, `09`, `10`, and `11`.
Written for both a first read-through and quick lookup later.

**`tree_id`** is the same 0-indexed number for a given physical tree
across every file in every stage — the join key throughout.

---

## Script 09 outputs — `outputs/las_analysis/`

### `las_attributes_per_file.csv`

One row per (file, attribute) combination.

| Column | Meaning |
|---|---|
| `file_label`, `file_path` | Which of the 4 raw LAS files this row describes |
| `attribute` | The LAS dimension name (e.g. `intensity`, `red`) |
| `description` | Plain-English explanation of what the attribute means |
| `n_unique_values` | How many distinct values appear — `1` means the field is a constant placeholder, never actually populated |
| `populated` | `True`/`False` shorthand for the above |
| `min_value`, `max_value` | Real range, when populated |

### `attribute_recommendations.csv`

One row per attribute (not per file): `attribute`, `recommendation`
(ESSENTIAL / USE / WORTH TESTING / NOT USEFUL / etc.), `reason`.

### `point_density_resolution.csv`

One row per condition (7 individual/merged combinations): `condition`,
`n_points`, `area_m2`, `density_pts_per_m2`, `avg_spacing_cm`,
`suggested_resolution_cm`. This is the analysis behind the resolution
table used throughout `06b`/`07b`/`08b`.

### `analysis_log.txt`

Full text transcript of everything printed to the console during the run.

---

## Script 06b outputs — `outputs/stage1_las_only/`

### `stage1_localized_trees_LAS_only.csv`

One row per tree (444 rows) — the master record of Stage 1's decision.

| Column | Meaning |
|---|---|
| `tree_id` | 0-indexed tree number |
| `orig_x`, `orig_y` | Original surveyed position from `trees_crop.shp` |
| `loc_x`, `loc_y` | Refined position Stage 1 decided is the real plant — what Stage 2/3 actually use |
| `method` | `localized` = a real candidate found and scored. `fallback_no_height_bump` = no height signal anywhere in the buffer. `fallback_zero_score` = a candidate existed but height and color signals never agreed on anything (combined score of zero everywhere). Fallback cases keep the original position unchanged. |
| `offset_m` | Distance moved from `orig_x/y`. Real result: median 0.050m, max 0.200m (the buffer edge) |
| `height_bump_m` | How much taller the winning pixel is than the buffer's own lowest point — relative, not absolute |
| `score` | Combined height×color×proximity score of the winner — meaningful only in comparison to other trees, not as an absolute number |
| `ambiguity_margin` | Gap between the winning score and the next-best genuinely different candidate (≥2 pixels away). Small = real ambiguity existed even if the absolute score looks fine |
| `used_rgb` | Whether real color data existed anywhere in this tree's buffer. Real result: 82.2% `True` |
| `review_recommended` | `True` for the bottom 20% of trees by `score` — a tested triage signal (not a perfect one) |
| `mean_intensity`, `pct_multireturn` | Diagnostic only — tested and found not to cleanly separate real plants from false positives (see script 09's findings), but recorded so you can inspect them yourself |

### `stage1_localized_points_LAS_only.shp` / `stage1_original_points_LAS_only.shp`

Point shapefiles at the refined and original positions respectively,
same field set: `tree_id`, `method`, `offset_m`, `height_m`, `score`,
`used_rgb`, `review`. CRS copied directly from your `trees_crop.prj`.

### `stage1_buffers_LAS_only.shp`

One circular polygon per tree — the search buffer's XY footprint (radius
= `buffer_radius_m`, 0.2m). This is a correct plan-view representation of
the cylinder's shape, since the real point selection is genuinely
unbounded in Z (see the README's Glossary for the full explanation).

### `stage1_chm_{condition}.tif` (7 files, one per condition)

Single-band height-above-ground, no color. `{condition}` is one of
`60m_flight_1`, `60m_flight_2`, `30m_flight_1`, `30m_flight_2`,
`60m_merged`, `30m_merged`, `all_4_merged`. Only the CHM for whichever
condition is set as `active_condition` in CONFIG drives the actual
localization decision above — the other 6 are for comparison only.

### `stage1_las_color_raster_{condition}.tif` (7 files)

The 3-tier visualization, per condition:
1. Real median color, where a colorized point exists in that cell
2. The cell's own **raw Z** value (not CHM), colormapped in viridis,
   where real points exist but none are colorized
3. Either a borrowed neighbor's data (`use_neighbor_borrow_fallback=True`,
   default) or plain black (`=False`), where zero points exist at all

Real confirmed numbers (`all_4_merged`): ~40% of cells fall into tier 2,
~12-27% (varies by condition's resolution) into tier 3.

---

## Script 07b outputs — `outputs/stage2_las_only/`

### `stage2_heights_all_conditions.csv`

One row per tree, height computed **7 separate ways**.

| Column pattern | Meaning |
|---|---|
| `height_{condition}` | Height (m) under that specific condition |
| `n_points_{condition}` | How many LiDAR points fell inside this tree's tight (0.12m) height-extraction buffer for that condition — **check this before trusting the height number next to it** |
| `low_confidence_{condition}` | `True` if `n_points_{condition}` is below `min_points_for_confidence` (5) |
| `mean_intensity_{condition}`, `pct_multireturn_{condition}` | Diagnostic only, same caveat as Stage 1's equivalent columns |

`{condition}` is each of the 7 values listed above. Real result summary in
the README's Stage 2 section.

### `merged_60m_merged.las`, `merged_30m_merged.las`, `merged_all_4_merged.las`

Actual merged point clouds you can open directly in CloudCompare — added
specifically so you can verify the point-level merge yourself, rather than
trust a script's claim about it. Confirmed via a grid-cell coverage check:
the merge is provably complete (zero cells present in a smaller set but
missing from the larger merged one).

### `chm_{condition}.tif` (7 files)

Same meaning as Stage 1's CHM files, rebuilt independently here (Stage 2
doesn't depend on Stage 1 having saved its own copies).

---

## Script 08b outputs — `outputs/stage3_las_only/`

### `stage3_crowns_LAS_only.csv`

One row per tree.

| Column | Meaning |
|---|---|
| `tree_id`, `x`, `y` | Tree identity and Stage 1's localized position |
| `crown_n_pixels` | How many CHM pixels the watershed assigned to this tree — **the single most important diagnostic column here** |
| `resolvable` | `True` only if `crown_n_pixels >= min_resolvable_pixels` (15). Real result: only ~25% of trees. Below this, a "crown" is 1-14 isolated pixels — a single LiDAR hit with no real 2D shape, not a bug, a genuine point-density limit (see README) |
| `crown_area_m2`, `crown_diameter_m` | Blank/NaN if not resolvable — deliberately not given a fabricated tiny number |
| `crown_max_height_m` | Tallest CHM value in this crown's pixels — computed even for unresolvable crowns, since even 1 pixel has *a* height |
| `eccentricity` | 0=circle, closer to 1=elongated line. Only computed if resolvable |
| `circularity` | `4π×area/perimeter²`, 1=circle, lower=irregular. Only computed if resolvable |
| `shape_ok` | `resolvable AND eccentricity < max_eccentricity` (0.9) — catches grass-blade-like artifacts even among pixel-count-passing crowns. Real result: 93-96% of resolvable crowns pass this too |

**For the most trustworthy subset, filter on `resolvable=True AND
shape_ok=True`.**

### `stage3_crowns_LAS_only.shp`

Same fields, abbreviated for the shapefile format: `resolv`, `shape_ok`,
`area_m2`, `diam_m`, `max_h_m`, `n_px`, `eccentr`, `circular`.

### `chm_used_for_segmentation.tif`

The exact CHM this run actually used — either reused directly from Stage
2's `chm_all_4_merged.tif` (if it existed) or built fresh with the same
memory-safe method. Saved so you always have a record of what was really
used, regardless of which code path ran.

---

## Script 10 outputs — `outputs/stage1_diagnostic/`

### `fallback_diagnostic_1.png`, `_2.png`, ... (one per 16 trees)

A grid of every tree where Stage 1's `method` was `fallback_zero_score`
or `fallback_no_height_bump`. Each panel: the LAS-derived RGB crop (gap-
filled, ~1.5cm cells matched to real point spacing), the original survey
point (red +), and the search buffer (yellow circle). No shapefiles or
CSVs — purely visual, meant for direct human review or to send back for
discussion.

---

## Script 11 outputs — `outputs/tif_verification/`

### `tif_verification_1.png`, `_2.png`, ... (one per 16 trees)

A grid sampling the **real drone photos** (not the LAS raster) at Stage
1's localized coordinates — much higher clarity than the LAS-derived
raster can achieve, confirmed to align with the true LiDAR coordinates to
within ~4cm in testing. Default samples 40 random localized trees
(`which_trees: "sample"`); can be set to `"all"` or `"localized_only"`
for a full review instead. Panels with a "no TIF coverage here" black
background reflect real photo-coverage gaps (confirmed: ~5% of trees
overall), not a bug.

---

## Quick-reference: column name patterns across the whole pipeline

- **`*_m` suffix**: a length in meters. **`*_m2`**: an area in m².
- **`n_*` / `*_n_pixels` / `n_points_*`**: a count, not a measurement —
  always check before trusting a nearby measurement column.
- **`{condition}` in a filename**: one of `60m_flight_1`, `60m_flight_2`,
  `30m_flight_1`, `30m_flight_2`, `60m_merged`, `30m_merged`,
  `all_4_merged` — the same 7 values everywhere in this project.
- **Boolean flags**: `used_rgb`, `review_recommended`, `resolvable`,
  `shape_ok`, `low_confidence_*` — filter on these for the trustworthy
  subset rather than treating every row as equally reliable.
- **`tree_id`**: the universal join key across every file, every stage.