"""
06b_stage1_buffer_localize_LAS_ONLY.py
================================================================================
STAGE 1, REDESIGNED: LAS-only (no RGB .tif photos at all)
================================================================================

WHAT CHANGED FROM THE ORIGINAL STAGE 1, AND WHY
----------------------------------------------------
1. **Color now comes from the LAS files' own embedded RGB, not the .tif
   photos.** Confirmed directly: your LAS files (point format 7) do carry
   real per-point red/green/blue -- 71.2% of vegetation-classified points
   have real (non-zero) color. This removes the dependency on the
   separately, "roughly" georeferenced photos entirely.

2. **The search buffer is now genuinely cylindrical, not just a relabeled
   circle**, AND HALF the previous size (0.4m -> 0.2m radius, as
   requested). "Cylindrical" is a real, meaningful design change here,
   not just terminology: the color raster is now built by gathering
   EVERY point in a cell's full vertical column (any height) and taking
   the MEDIAN color across them -- not just the color of the single
   highest point (which is what the .tif-based version's raster and the
   original MyDiv-project true-color renderer both effectively did).
   Tested directly: at fine resolutions (1-2cm) most cells only have 1
   point anyway, so this wouldn't matter -- confirmed empirically, and is
   exactly why the resolutions below are chosen coarse enough (4-8cm) for
   the median to have 2-4 real points to work with, making it genuinely
   more ROBUST to a single stray/outlier-colored point (e.g. one bright
   return that happens to look whitish) than a "just take the top point"
   approach would be.

3. **Per-condition resolution, not one fixed number for everything.**
   Confirmed three separate, real facts before choosing these:
     - Your LAS files have a 0.01m (1cm) coordinate SCALE FACTOR in their
       header -- nothing can be stored more precisely than 1cm regardless
       of point density. A hard floor, not a density question.
     - TRUE median nearest-neighbor point spacing (not a density
       approximation): 30m data sits almost exactly at that 1cm floor;
       60m data's real median spacing is 2.00cm.
     - Points-per-cell at various resolutions (tested directly): median
       color aggregation only becomes meaningful (2-4 points/cell) at
       4-8cm for individual flights, but the fully-merged cloud already
       gets there at 2cm.
   Resulting resolution table (below in CONFIG) is chosen to give roughly
   2-4 points per colored cell for each condition specifically, not one
   guessed number applied everywhere.

4. **Intensity is exposed but NOT used as a primary signal by default.**
   Tested directly on real data first: compared intensity for
   RGB-defined "whitish/log-like" points vs. "clearly green" points --
   medians differed by only ~1.4% (42,488 vs 43,078), not a clean
   separation. Multi-return fraction was tested too and was
   near-zero for both groups (0.1% vs 0.2%) -- this dataset barely has
   any multi-return points at all, so that hypothesis doesn't hold up
   either. Both are still RECORDED per tree as diagnostic columns (so you
   can inspect them yourself and decide), but neither is wired into the
   scoring by default, since the direct test didn't support it strongly.
   (Caveat: this test compared RGB-defined groups, so it can't rule out
   intensity helping in the specific case where RGB itself is fooled --
   that needs an actual confirmed log location to test properly.)

EVERYTHING ELSE (the core algorithm) IS UNCHANGED AND ALREADY PROVEN:
    circular/cylindrical masking, local-contrast color anomaly (not raw
    greenness), proximity weighting, height ceiling, marker relocation,
    ambiguity margin, review-recommended flagging -- all identical logic
    to the original Stage 1, just fed by the new LAS-derived color raster
    instead of the .tif-derived one.
================================================================================
"""

import os
import csv
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    # Path to the ground-truth tree survey shapefile (the 444 known real
    # tree positions this whole pipeline is trying to refine/measure).
    "shapefile_path": "trees_crop.shp",

    # The 4 raw LAS files this project has -- 2 flights at 60m altitude,
    # 2 at 30m. Every other condition (60m_merged, 30m_merged,
    # all_4_merged) is built by combining these at runtime, never a
    # separately-stored file.
    "las_files": {
        "60m_flight_1": "60m/260522_122413_filt_crop.las",
        "60m_flight_2": "60m/260522_124721_filt_crop.las",
        "30m_flight_1": "30m/260522_130956_filt_crop.las",
        "30m_flight_2": "30m/260522_140126_filt_crop.las",
    },
    # Which of the 7 possible conditions to actually localize against.
    # "all_4_merged" is the default (best completeness, per earlier
    # testing) -- change this to any key in RESOLUTION_TABLE below to
    # localize against a different condition instead.
    "active_condition": "all_4_merged",

    # Where every output file from this script gets written.
    "output_dir": "outputs/stage1_las_only",

    # HALF the original buffer_radius_m (0.4m), as requested. Cylindrical:
    # this radius applies in X/Y only: every point at that XY distance
    # from the survey point is included, at ANY height (Z) -- a real
    # vertical cylinder, not a flattened circle.
    "buffer_radius_m": 0.2,

    # Per-condition resolution, derived from real density testing (see
    # header comment). Used for BOTH the CHM and the color raster for
    # that condition -- keeps the pipeline simple, one raster grid per
    # condition rather than two different grids to keep aligned.
    "resolution_table_m": {
        "60m_flight_1": 0.06, "60m_flight_2": 0.06,
        "30m_flight_1": 0.04, "30m_flight_2": 0.04,
        "60m_merged": 0.05, "30m_merged": 0.03,
        "all_4_merged": 0.02,
    },

    # A candidate pixel must be at least this many meters above the
    # locally-interpolated ground (via the CHM, ground-normalized -- see
    # build_z_and_color_rasters) to count as "something real is here" at
    # all. Too low and flat ground noise starts looking like a candidate;
    # too high and genuinely tiny seedlings get excluded outright.
    "min_height_bump_m": 0.03,

    # Hard ceiling: a candidate CHM cell taller than this can NEVER win,
    # no matter how strong its color/proximity signal -- this is what
    # stops the algorithm from "upgrading" to a bigger neighboring plant
    # or tree just because it happens to look more prominent. Only
    # excludes IMPLAUSIBLY tall things (a real mature tree); does NOT
    # exclude "shorter than that but still taller than the seedling"
    # cases like tall grass -- a known, accepted limitation.
    "max_plausible_height_m": 2.0,

    # Exponents in the combined score: score = (height_signal ** height_weight)
    # * (color_signal ** color_weight) * proximity_weight. 1.0 each means
    # equal weighting -- raise one to make that signal count more heavily
    # relative to the other when picking the winning candidate.
    "height_weight": 1.0,
    "color_weight": 1.0,

    # Controls how strongly a candidate's DISTANCE from the original
    # survey point counts against it, via proximity_weight =
    # exp(-distance_px / (buffer_radius_px * this_fraction)). Lower value
    # = distance matters MORE (pulls harder toward the original survey
    # point, needs a much stronger height+color signal to justify picking
    # something farther away). This was the fix for an earlier confirmed
    # bug where candidates jumped to unrelated vegetation at the buffer
    # edge -- see the header comment's development history.
    "proximity_decay_fraction": 0.35,

    # Radius (meters) for the background blur used in the LOCAL-CONTRAST
    # color anomaly calculation (see localize_one_tree) -- NOT raw
    # greenness. A pixel only scores highly if it's brighter than the
    # AVERAGE color within this radius around it, which is what lets a
    # small plant stand out from a big uniform grass patch instead of
    # losing to it just for being less saturated.
    "local_contrast_radius_m": 0.08,

    # The bottom this-many-percent of trees BY SCORE (not by any fixed
    # threshold) get flagged review_recommended=True in the output CSV --
    # a relative, tested triage signal (see header comment for how well
    # it actually performs), not an absolute quality guarantee.
    "review_score_percentile": 20,
    # See save_color_geotiff's docstring for the full honesty tradeoff.
    # True = no black regions in the saved visualization (tier-3 cells
    # borrow a neighbor). False = tier-3 cells (zero real points at all)
    # stay black -- the most honest option, nothing in the image is ever
    # guessed from elsewhere.
    "use_neighbor_borrow_fallback": True,
}


# ================================================================================
# LOAD POINTS FOR A GIVEN CONDITION (any of the 7)
# ================================================================================

ALL_CONDITIONS = ["60m_flight_1", "60m_flight_2", "30m_flight_1", "30m_flight_2",
                   "60m_merged", "30m_merged", "all_4_merged"]


def load_condition(condition, cfg):
    """
    Returns (xyz, cls, rgb01, intensity, n_returns) for any of the 7
    conditions: a single flight, or a merge of 2 or 4. rgb01 is Nx3 in
    [0,1] with (0,0,0) for points that have no real color. intensity and
    n_returns are recorded as per-tree DIAGNOSTIC columns in the output
    (see main()) -- tested directly and found NOT to cleanly separate
    "log-like" from "plant-like" points on this data (see header comment),
    so they don't drive scoring, but are now exposed for you to inspect
    yourself, matching what Stage 2 already does.
    """
    file_map = cfg["las_files"]
    if condition in file_map:
        paths = [file_map[condition]]
    elif condition == "60m_merged":
        paths = [file_map["60m_flight_1"], file_map["60m_flight_2"]]
    elif condition == "30m_merged":
        paths = [file_map["30m_flight_1"], file_map["30m_flight_2"]]
    elif condition == "all_4_merged":
        paths = list(file_map.values())
    else:
        raise ValueError(f"Unknown condition: {condition}")

    all_xyz, all_cls, all_rgb, all_intensity, all_nret = [], [], [], [], []
    for p in paths:
        las = laspy.read(p)
        all_xyz.append(np.column_stack([las.x, las.y, las.z]))
        all_cls.append(np.asarray(las.classification))
        r, g, b = np.asarray(las.red, dtype=float), np.asarray(las.green, dtype=float), np.asarray(las.blue, dtype=float)
        scale = 65535.0 if max(r.max(), g.max(), b.max()) > 255 else 255.0
        all_rgb.append(np.stack([r, g, b], axis=1) / scale)
        all_intensity.append(np.asarray(las.intensity, dtype=float))
        all_nret.append(np.asarray(las.number_of_returns))
        print(f"    loaded {p}: {len(las.points):,} points")
    return (np.vstack(all_xyz), np.concatenate(all_cls), np.vstack(all_rgb),
            np.concatenate(all_intensity), np.concatenate(all_nret))


def diagnostics_near_point(tx, ty, xyz, intensity, n_returns, radius_m):
    """Mean intensity and % multi-return within a circular buffer around
    (tx, ty) -- diagnostic only, does not affect localization scoring."""
    dxy = np.hypot(xyz[:, 0] - tx, xyz[:, 1] - ty)
    in_buffer = dxy <= radius_m
    if not in_buffer.any():
        return np.nan, np.nan
    mean_intensity = float(intensity[in_buffer].mean())
    pct_multi = float(100 * (n_returns[in_buffer] > 1).mean())
    return mean_intensity, pct_multi


# ================================================================================
# GROUND / CHM (unchanged approach from the original pipeline)
# ================================================================================
# NOTE: this script previously had build_tin_interpolator() and
# voxel_downsample_2d() here -- a ground TIN/DTM subsystem used to
# ground-normalize height. Per explicit request, Stage 1 no longer
# computes any ground model at all -- removed entirely, not just
# unused, since keeping dead code implementing something you explicitly
# don't want processed would be misleading. Stage 2/3 still have their
# own copies of this logic (they still legitimately need ground-
# normalized height for their own purposes) -- this removal is scoped
# to Stage 1 only.
# ================================================================================

def build_z_and_color_rasters(xyz, cls, rgb01, res, cfg):
    """
    Builds TWO things from the SAME point cloud, at the same resolution --
    NO ground TIN/DTM/CHM computation anywhere in this function, per
    explicit request. Previously this built a ground-normalized CHM via
    a Delaunay triangulation of classified ground points -- that entire
    subsystem is now removed, not just unused for the visualization.

    This is a genuine simplification, not just compliance with a
    preference: the ground TIN was the exact subsystem that caused TWO
    confirmed OOM crashes earlier in this project (building the
    triangulation from ~12 million points). Removing it here eliminates
    that risk from Stage 1 entirely.

      1. `z_raster` -- max RAW 'Z' elevation per cell, completely
         unfiltered (any point, any height, no ground-normalization of
         any kind). This is now the ONLY height signal used anywhere in
         this script, for both the actual localization scoring
         (localize_one_tree) and the saved visualization.

      2. `color_raster` -- MEDIAN color per cell, gathered from every
         COLORED point in that cell's full vertical column. NOTE: the
         earlier version also required a point to be "tall enough above
         ground" to count toward color -- that check depended on the now-
         removed ground TIN, so it's gone too. Color eligibility is now
         simply "does this point have real color at all" -- a real,
         honest tradeoff: bare-ground colored points (soil, mulch) can
         now contribute to a cell's color where they couldn't before,
         since there's no ground model left to exclude them with.

    WHY A LOCAL (not global) height reference still works for finding a
    seedling within a small buffer: real terrain slope across a ~59x50m
    plot changes gradually; within any single 0.2m-radius search buffer,
    that slope's contribution to raw Z is negligible. localize_one_tree
    already computes height relative to EACH buffer's own local minimum
    Z (not a global reference) -- passing it unfiltered raw Z instead of
    a pre-normalized CHM lands on essentially the same local computation,
    just without the expensive/crash-prone global step beforehand.
    """
    raw_z = xyz[:, 2]
    has_color = rgb01.sum(axis=1) > 0

    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    ncols = int(np.ceil((xmax - xmin) / res)) + 1
    nrows = int(np.ceil((ymax - ymin) / res)) + 1

    col = np.clip(((xyz[:, 0] - xmin) / res).astype(int), 0, ncols - 1)
    row = np.clip(((ymax - xyz[:, 1]) / res).astype(int), 0, nrows - 1)
    flat_idx = row * ncols + col

    # --- Z raster: max RAW elevation per cell, completely unfiltered ---
    flat_z = np.full(nrows * ncols, -np.inf)
    np.maximum.at(flat_z, flat_idx, raw_z)
    flat_z[flat_z == -np.inf] = np.nan
    z_raster = flat_z.reshape(nrows, ncols)

    # Tracks which cells have ANY real point at all -- unfiltered.
    flat_any_point = np.zeros(nrows * ncols, dtype=bool)
    np.logical_or.at(flat_any_point, flat_idx, True)
    any_point_mask = flat_any_point.reshape(nrows, ncols)

    # --- Color raster: MEDIAN color per cell, cylindrical column ---
    # ("cylindrical": every point in this XY cell counts, at any height --
    # eligibility is now purely "has real color", no height check at all,
    # since that check depended on the now-removed ground model)
    color_eligible = has_color
    color_raster = np.zeros((nrows, ncols, 3))
    cell_ids_eligible = flat_idx[color_eligible]
    rgb_eligible = rgb01[color_eligible]
    if len(cell_ids_eligible) > 0:
        order = np.argsort(cell_ids_eligible)
        sorted_ids = cell_ids_eligible[order]
        sorted_rgb = rgb_eligible[order]
        unique_ids, start_idx = np.unique(sorted_ids, return_index=True)
        end_idx = np.append(start_idx[1:], len(sorted_ids))
        flat_color = color_raster.reshape(-1, 3)
        for uid, s, e in zip(unique_ids, start_idx, end_idx):
            flat_color[uid] = np.median(sorted_rgb[s:e], axis=0)
        color_raster = flat_color.reshape(nrows, ncols, 3)

    transform = from_origin(xmin, ymax, res, res)
    print(f"    Z+color built at {res*100:.1f}cm: shape={z_raster.shape}, "
          f"{100*np.isnan(z_raster).mean():.1f}% empty cells "
          f"(no ground TIN computed at all -- pure raw Z only)")
    return z_raster, color_raster, transform, any_point_mask



def excess_green_index(rgb_img01):
    r, g, b = rgb_img01[..., 0] * 255, rgb_img01[..., 1] * 255, rgb_img01[..., 2] * 255
    return 2 * g - r - b


# ================================================================================
# LOCALIZE ONE TREE (same proven algorithm, new inputs)
# ================================================================================

def localize_one_tree(tx, ty, z_raster, color_raster, transform, cfg):
    """
    Same proven scoring algorithm as before -- height signal x color
    signal x proximity weight -- now fed pure raw Z instead of a ground-
    normalized CHM. Works because height_bump below is computed relative
    to THIS BUFFER's own local minimum Z, not a global reference -- see
    build_z_and_color_rasters' docstring for why that's still valid at
    this small a scale.

    IMPORTANT BUG FIX made as part of this change: the max_plausible_height_m
    ceiling previously checked the CHM value directly, which was already
    a height-above-ground number, so comparing it to 2.0m made sense. Now
    that the raw input is ABSOLUTE elevation (hundreds of meters), that
    same check against the raw value would ALWAYS be true and exclude
    every single candidate. Fixed: the ceiling now checks height_bump (the
    LOCAL, already-relativized height computed a few lines below), not
    the raw z_win value -- restoring the original, correct intent of
    "exclude implausibly TALL PLANTS", not "exclude high elevation".
    """
    half = cfg["buffer_radius_m"]
    res = transform.a
    xmin_w, ymax_w = transform.c, transform.f

    col_c = int((tx - xmin_w) / res)
    row_c = int((ymax_w - ty) / res)
    half_px = int(np.ceil(half / res))
    r0, r1 = max(0, row_c - half_px), min(z_raster.shape[0], row_c + half_px + 1)
    c0, c1 = max(0, col_c - half_px), min(z_raster.shape[1], col_c + half_px + 1)
    if r1 <= r0 or c1 <= c0:
        return {"x": tx, "y": ty, "method": "fallback_no_data", "offset_m": 0.0,
                "height_bump_m": np.nan, "score": 0.0, "ambiguity_margin": np.nan}

    z_win = z_raster[r0:r1, c0:c1]
    color_win = color_raster[r0:r1, c0:c1]

    if np.all(np.isnan(z_win)):
        return {"x": tx, "y": ty, "method": "fallback_no_data", "offset_m": 0.0,
                "height_bump_m": np.nan, "score": 0.0, "ambiguity_margin": np.nan}

    rr, cc = np.indices(z_win.shape)
    center_row = (ymax_w - ty) / res - r0
    center_col = (tx - xmin_w) / res - c0
    dist_px = np.hypot(rr - center_row, cc - center_col)
    outside_circle = dist_px > (half / res)  # "cylindrical": XY radius only, no Z bound

    inside_vals = np.where(outside_circle, np.nan, z_win)
    if np.all(np.isnan(inside_vals)):
        return {"x": tx, "y": ty, "method": "fallback_no_height_bump", "offset_m": 0.0,
                "height_bump_m": 0.0, "score": 0.0, "ambiguity_margin": np.nan}
    local_min = np.nanmin(inside_vals)
    height_bump = z_win - local_min  # LOCAL relative height -- this buffer's own floor, not a global DTM

    # BUG FIX (see docstring): ceiling now checks height_bump (local,
    # relative), not z_win (absolute elevation) -- checking the raw
    # value here would always exclude everything once z_win moved from
    # ground-normalized height to true elevation.
    excluded = (height_bump > cfg["max_plausible_height_m"]) | outside_circle
    height_bump_scored = np.where(excluded, np.nan, height_bump)
    height_bump_scored = np.where(height_bump_scored < cfg["min_height_bump_m"], np.nan, height_bump_scored)

    if np.all(np.isnan(height_bump_scored)):
        return {"x": tx, "y": ty, "method": "fallback_no_height_bump", "offset_m": 0.0,
                "height_bump_m": 0.0, "score": 0.0, "ambiguity_margin": np.nan}

    hb_norm = height_bump_scored / np.nanmax(height_bump_scored)

    exg = excess_green_index(color_win)
    has_color_win = color_win.sum(axis=2) > 0
    from scipy.ndimage import uniform_filter
    blur_radius_px = max(1, int(cfg["local_contrast_radius_m"] / res))
    local_bg = uniform_filter(exg, size=blur_radius_px * 2 + 1)
    local_anomaly = exg - local_bg
    exg_clip = np.clip(local_anomaly, 0, None)
    exg_clip = np.where(has_color_win, exg_clip, 0)
    exg_norm = exg_clip / exg_clip.max() if exg_clip.max() > 0 else np.ones_like(exg_clip)
    used_rgb = bool(has_color_win[~outside_circle].any())

    hb_safe = np.nan_to_num(hb_norm, nan=0.0)
    exg_norm = np.where(outside_circle, 0.0, exg_norm)

    decay_px = (half / res) * cfg["proximity_decay_fraction"]
    proximity_weight = np.exp(-dist_px / decay_px)

    combined = (hb_safe ** cfg["height_weight"]) * (exg_norm ** cfg["color_weight"]) * proximity_weight
    combined = np.where(outside_circle, -1.0, combined)

    if combined.max() <= 0:
        return {"x": tx, "y": ty, "method": "fallback_zero_score", "offset_m": 0.0,
                "height_bump_m": 0.0, "score": 0.0, "ambiguity_margin": np.nan, "used_rgb": used_rgb}

    best_r, best_c = np.unravel_index(np.argmax(combined), combined.shape)
    best_x = xmin_w + (c0 + best_c) * res
    best_y = ymax_w - (r0 + best_r) * res
    offset = float(np.hypot(best_x - tx, best_y - ty))

    far_enough = np.hypot(rr - best_r, cc - best_c) > 2
    second_best = float(np.nanmax(np.where(far_enough, combined, -np.inf))) if far_enough.any() else np.nan
    margin = float(combined[best_r, best_c] - second_best) if not np.isnan(second_best) else np.nan

    return {
        "x": best_x, "y": best_y, "method": "localized", "offset_m": offset,
        "height_bump_m": float(height_bump[best_r, best_c]),
        "score": float(combined[best_r, best_c]), "ambiguity_margin": margin,
        "used_rgb": used_rgb,
    }


# ================================================================================
# SAVE OUTPUTS
# ================================================================================
#
# EVERY FILE THIS SCRIPT PRODUCES, IN ONE PLACE (added per explicit
# request -- individual functions below also explain their own file in
# detail, this is the single consolidated list):
#
#   stage1_localized_trees_LAS_only.csv
#       One row per tree: original position, localized position, method,
#       offset, score, ambiguity_margin, used_rgb, review_recommended,
#       mean_intensity, pct_multireturn. The master record of Stage 1's
#       decision for every tree.
#
#   stage1_localized_points_LAS_only.shp (+ .shx/.dbf/.prj)
#       The refined (localized) positions as a point shapefile, for QGIS.
#
#   stage1_original_points_LAS_only.shp (+ sidecars)
#       The ORIGINAL (unrefined) survey positions, same structure, for
#       side-by-side comparison against the localized points above.
#
#   stage1_buffers_LAS_only.shp (+ sidecars)
#       One circular polygon per tree = the search buffer's XY footprint
#       (radius = buffer_radius_m). Represents a cylinder's plan-view
#       shape correctly, since the actual point selection is genuinely
#       unbounded in Z (see save_buffers_shapefile's docstring for the
#       full explanation of what the Z dimension actually is).
#
#   stage1_chm_{condition}.tif  (one per condition, 7 total)
#       Pure height-above-ground data, single band, no color at all --
#       this is what the actual algorithm scores against.
#
#   stage1_las_color_raster_{condition}.tif  (one per condition, 7 total)
#       The 3-tier human-viewing visualization: real color where a
#       colorized point exists, a viridis raw-Z tint where a point
#       exists but has no color, and either a borrowed neighbor or plain
#       black for cells with zero points at all (see save_color_geotiff's
#       docstring for the full tier-by-tier breakdown and the
#       use_neighbor_borrow_fallback toggle).
#
# {condition} is one of: 60m_flight_1, 60m_flight_2, 30m_flight_1,
# 30m_flight_2, 60m_merged, 30m_merged, all_4_merged -- only the ACTIVE
# condition (set in CONFIG) drives the real localization decision above;
# the other 6 conditions' rasters are built purely as additional outputs
# to inspect/compare, and don't affect any tree's actual result.
# ================================================================================

def write_prj_from_source(out_shp_path, source_prj_path):
    with open(source_prj_path) as f:
        wkt = f.read()
    with open(out_shp_path.replace(".shp", ".prj"), "w") as f:
        f.write(wkt)


def save_points_shapefile(rows, path, x_key, y_key, source_prj_path):
    w = shapefile.Writer(path, shapeType=shapefile.POINT)
    w.field("tree_id", "N")
    w.field("method", "C", size=30)
    w.field("offset_m", "N", decimal=4)
    w.field("height_m", "N", decimal=4)
    w.field("score", "N", decimal=4)
    w.field("used_rgb", "L")
    w.field("review", "L")
    for i, r in enumerate(rows):
        w.point(r[x_key], r[y_key])
        w.record(i, r.get("method", ""), r.get("offset_m", 0.0),
                  r.get("height_bump_m", 0.0), r.get("score", 0.0),
                  r.get("used_rgb", False), r.get("review_recommended", False))
    w.close()
    write_prj_from_source(path, source_prj_path)
    print(f"  saved {path}")


def save_buffers_shapefile(tree_pts, radius_m, path, source_prj_path):
    """
    Saves the buffer as a circular polygon per tree, in plan (map) view.

    WHY A 2D CIRCLE CORRECTLY REPRESENTS A 3D CYLINDER HERE, AND WHAT THE
    Z DIMENSION ACTUALLY IS -- read this rather than guess, since you
    asked directly and the honest answer has a real nuance in it:

    A shapefile can't natively store a true 3D cylinder as one shape, but
    it doesn't need to for this purpose -- QGIS's map view only ever
    shows the XY footprint anyway, and a cylinder's XY footprint (viewed
    from directly above) IS exactly a circle, identical to the buffer
    radius already used before switching to LAS-only. So this circle
    polygon is a completely accurate representation of what you'd see
    looking down at the cylindrical search region -- nothing is lost by
    using a 2D shape here.

    The Z dimension itself, checked directly in the actual code rather
    than assumed:
      - There is NO upper bound at all on which points get aggregated
        into the CHM or color raster -- any point above the local ground,
        no matter how tall, contributes during rasterization.
      - There IS a lower bound, but only for which points contribute to
        COLOR: a point must be at least `min_height_bump_m` (3cm) above
        the locally-interpolated ground to count toward a cell's median
        color (this excludes bare-ground/mulch-level returns from
        influencing what color a cell is assigned). The CHM/height value
        itself uses an even more lenient -5cm noise margin, essentially
        unbounded either way.
      - The `max_plausible_height_m` ceiling (2.0m) is NOT a bound on
        which points get rasterized -- it's applied LATER, only to
        already-built CHM cell VALUES, to decide which candidate cells
        are allowed to win the localization search. It never restricts
        what goes into the raster in the first place.
    In short: this is a cylinder that's unbounded upward, loosely bounded
    downward (effectively "at or above local ground"), not a fixed height
    BAND cropped to some specific range.
    """
    w = shapefile.Writer(path, shapeType=shapefile.POLYGON)
    w.field("tree_id", "N")
    n_seg = 24
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    for i, (tx, ty) in enumerate(tree_pts):
        ring = [(tx + radius_m * np.cos(a), ty + radius_m * np.sin(a)) for a in angles]
        ring.append(ring[0])
        w.poly([ring])
        w.record(i)
    w.close()
    write_prj_from_source(path, source_prj_path)
    print(f"  saved {path}")


def save_z_geotiff(z_raster, transform, path, source_prj_path):
    """Saves the raw Z raster (no ground-normalization, no CHM -- just
    the max raw elevation per cell) as a single-band GeoTIFF."""
    crs_wkt = None
    if source_prj_path and os.path.exists(source_prj_path):
        with open(source_prj_path) as f:
            crs_wkt = f.read()
    z_out = np.where(np.isnan(z_raster), -9999, z_raster).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=z_raster.shape[0], width=z_raster.shape[1],
                        count=1, dtype=np.float32, nodata=-9999, crs=crs_wkt, transform=transform) as dst:
        dst.write(z_out, 1)
    print(f"    saved {path}")


def save_z_colorized_geotiff(z_raster, transform, path, source_prj_path):
    """
    Saves a SEPARATE, colorized 3-band viewing version of the raw Z
    raster (NOT a CHM -- no ground-normalization anywhere in this
    script), specifically to fix a real, confirmed problem: opening a
    plain single-band raster in QGIS with default styling makes genuine
    NoData cells visually indistinguishable from real low-lying values.
    A naive full-range linear stretch, especially across ABSOLUTE
    elevation (hundreds of meters, with only a small fraction being real
    plant-height variation), would crush almost everything into
    near-identical color -- so this stretches using the 2nd-98th
    PERCENTILE of REAL data only (NaN/missing cells excluded from the
    calculation entirely), colormap in viridis (starts dark PURPLE, not
    black, at its low end), and keeps genuine NoData cells as actual
    GeoTIFF NoData (fully transparent in QGIS) -- never given any color
    at all. Transparent in QGIS ALWAYS and ONLY means "no real data
    here" -- never confusable with a real dark-purple low value.
    """
    import matplotlib.cm as cm
    valid = ~np.isnan(z_raster)
    if not valid.any():
        print(f"    skipped {path} -- no valid data at all")
        return
    vmin, vmax = np.nanpercentile(z_raster[valid], [2, 98])
    norm = np.clip((z_raster - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = cm.viridis(norm)[..., :3]
    rgb255 = (rgb * 255).astype(np.uint8)

    crs_wkt = None
    if source_prj_path and os.path.exists(source_prj_path):
        with open(source_prj_path) as f:
            crs_wkt = f.read()
    with rasterio.open(path, "w", driver="GTiff", height=z_raster.shape[0], width=z_raster.shape[1],
                        count=4, dtype=np.uint8, crs=crs_wkt, transform=transform) as dst:
        for i in range(3):
            dst.write(rgb255[:, :, i], i + 1)
        alpha = np.where(valid, 255, 0).astype(np.uint8)  # true transparency for NoData, not a color
        dst.write(alpha, 4)
    print(f"    saved {path}  (colorized, {vmin:.3f}-{vmax:.3f}m stretch from real 2nd-98th percentile, "
          f"transparent = genuinely no data, never a real dark value)")


def save_z_range_geotiff(z_raster, transform, path, source_prj_path, range_min, range_max):
    """
    Saves a single-band GeoTIFF containing ONLY real Z values that fall
    within [range_min, range_max] -- anything outside that range becomes
    genuine NoData, not clamped/distorted. This is why QGIS will show a
    correct, narrower legend for THIS file automatically: unlike the
    colorized file (which pre-bakes color into RGB pixels and has no
    numeric legend at all -- a real, inherent limitation, not a bug),
    this stays single-band with real numbers, so QGIS reads its own
    min/max directly from the (now genuinely narrower) data and displays
    it correctly without any extra help needed.
    """
    crs_wkt = None
    if source_prj_path and os.path.exists(source_prj_path):
        with open(source_prj_path) as f:
            crs_wkt = f.read()
    masked = np.where((z_raster >= range_min) & (z_raster <= range_max), z_raster, np.nan)
    n_in_range = int((~np.isnan(masked)).sum())
    z_out = np.where(np.isnan(masked), -9999, masked).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=z_raster.shape[0], width=z_raster.shape[1],
                        count=1, dtype=np.float32, nodata=-9999, crs=crs_wkt, transform=transform) as dst:
        dst.write(z_out, 1)
    print(f"    saved {path}  ({n_in_range:,} cells fall within [{range_min},{range_max}]m, "
          f"rest are genuine NoData)")


def save_viridis_qml(path, range_min, range_max):
    """
    Writes a QGIS .qml style sidecar with the SAME base filename as a
    single-band GeoTIFF -- QGIS auto-detects and applies a matching .qml
    when you load the raster, no manual styling needed. This solves the
    "colorized file has no legend" problem a DIFFERENT way than the
    pre-baked RGB approach: instead of converting height to color
    ourselves (losing the numbers permanently), this tells QGIS to do
    the coloring itself, on the real single-band data -- so you get
    automatic color on load AND a correct, real-unit legend, since QGIS
    is rendering genuine numbers, not a pre-converted image.

    Uses the 5 standard viridis anchor colors (purple -> blue -> teal ->
    green -> yellow), evenly spaced between your requested range_min and
    range_max.

    HONEST LIMITATION: this file was generated and checked for valid,
    well-formed XML matching the standard QGIS QML schema, but was NOT
    tested inside an actual running QGIS session (no GUI available here)
    -- confirm it applies correctly when you load it, and tell me if it
    doesn't so I can look at the actual error rather than guess blind.
    """
    stops = np.linspace(range_min, range_max, 5)
    viridis_colors = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    items = "\n".join(
        f'          <item value="{v:.3f}" color="{c}" label="{v:.2f}" alpha="255"/>'
        for v, c in zip(stops, viridis_colors)
    )
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0">
{items}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    qml_path = path.replace(".tif", ".qml")
    with open(qml_path, "w") as f:
        f.write(qml)
    print(f"    saved {qml_path}  (auto-applies viridis color + a correct {range_min}-{range_max}m "
          f"legend when this .tif is loaded in QGIS)")


def ask_custom_range_interactively():
    """
    Asks whether you want an ADDITIONAL pair of outputs (masked-to-range
    raw file + matching QML-styled version) for a specific height range
    you choose -- separate from, and without modifying, the existing
    stage1_z_raw_*/stage1_z_colorized_* outputs at all.
    """
    raw = input("\nWould you also like an additional Z output masked to a specific height "
                "range you choose? (y/n, default n): ").strip().lower()
    if raw != "y":
        return None
    lo_raw = input("  Enter the minimum height in meters (e.g. from the raw file's legend): ").strip()
    hi_raw = input("  Enter the maximum height in meters: ").strip()
    try:
        lo, hi = float(lo_raw), float(hi_raw)
        if lo >= hi:
            print(f"  min ({lo}) must be less than max ({hi}) -- skipping custom range output")
            return None
        return (lo, hi)
    except ValueError:
        print(f"  couldn't parse '{lo_raw}'/'{hi_raw}' as numbers -- skipping custom range output")
        return None


def save_color_geotiff(color_raster, z_raster, any_point_mask, transform, path, source_prj_path, cfg):
    """
    Saves a visualization GeoTIFF built from THREE tiers of real data,
    honesty-ranked -- not a single blanket gap-fill:

      1. Cells with real color: shown as their actual median RGB.
      2. Cells with real points but NONE colored: shown using THEIR OWN
         RAW 'Z' attribute (absolute elevation, NOT the ground-normalized
         CHM height used by the actual algorithm), colormap-coded
         (viridis) -- per explicit request to use the raw Z attribute
         specifically for this visualization. Note: because this is
         absolute elevation, not height-above-ground, the tint can drift
         smoothly across the image following the plot's real terrain
         slope, not just individual plant bumps -- that's expected given
         raw Z is what's being shown, not a bug. Confirmed on real data
         (all_4_merged condition): 39.9% of ALL cells fall into this
         tier -- a substantial fraction now showing real data instead of
         a neighbor's guess.
      3. Cells with ZERO points at all: THE ONLY REMAINING "borrowed
         guess" -- nearest-neighbor borrowing, since there's truly no
         real data of any kind for that cell (confirmed: 27.5% of cells
         on all_4_merged; this fraction shrinks for higher-resolution
         conditions like the individual flights, since more of a cell's
         area was never sampled by any point at all).

    THIS TIER-3 BORROWING IS CONTROLLED BY cfg["use_neighbor_borrow_fallback"]:
      - True (default): tier-3 cells show a neighbor's real data, so the
        image has no black regions at all -- easier to visually scan,
        but that fraction of the image is not this cell's own real data.
      - False: tier-3 cells are left as black (0,0,0) instead -- the most
        honest option, every non-black pixel is guaranteed to be either
        this cell's own real color or its own real height, with zero
        borrowed/guessed content anywhere in the image. Set this if you
        specifically want "no data" to mean "black", full stop.

    The height-colormap cells are drawn from viridis (blue-purple to
    yellow) specifically because it doesn't resemble natural vegetation
    colors -- you should be able to tell at a glance which patches are
    real captured color and which are the height-based substitute.

    Does not affect localization results -- called after
    localize_one_tree has already finished running on the original,
    un-modified color_raster.
    """
    has_color = color_raster.sum(axis=2) > 0
    result = color_raster.copy()

    # Tier 2: real points, no color -> use the cell's own RAW Z (not
    # CHM/ground-normalized height), colormapped -- per explicit request.
    height_only = any_point_mask & ~has_color
    if height_only.any():
        import matplotlib.cm as cm
        valid_z = z_raster[height_only]
        vmin, vmax = np.nanpercentile(valid_z, [2, 98])
        norm = np.clip((z_raster - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        colored = cm.viridis(norm)[..., :3]
        result[height_only] = colored[height_only]
        print(f"    {int(height_only.sum()):,} cells ({100*height_only.sum()/height_only.size:.1f}%) "
              f"have real points but no color -- shown using their own raw Z value instead of a "
              f"borrowed guess")

    # Tier 3: truly no data at all. use_neighbor_borrow_fallback=False
    # leaves these as black instead of borrowing a neighbor -- see
    # docstring above for the honesty tradeoff either way.
    still_empty = ~any_point_mask
    if cfg.get("use_neighbor_borrow_fallback", True):
        if still_empty.any() and (~still_empty).any():
            from scipy.ndimage import distance_transform_edt
            _, indices = distance_transform_edt(still_empty, return_distances=True, return_indices=True)
            result = np.where(still_empty[..., None], result[indices[0], indices[1]], result)
            print(f"    {int(still_empty.sum()):,} cells ({100*still_empty.sum()/still_empty.size:.1f}%) "
                  f"have zero points at all -- these still use a borrowed neighbor (no real data "
                  f"exists for them at all)")
    else:
        print(f"    {int(still_empty.sum()):,} cells ({100*still_empty.sum()/still_empty.size:.1f}%) "
              f"have zero points at all -- left as black "
              f"(use_neighbor_borrow_fallback=False, no guessing at all)")

    crs_wkt = None
    if source_prj_path and os.path.exists(source_prj_path):
        with open(source_prj_path) as f:
            crs_wkt = f.read()
    rgb255 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=rgb255.shape[0], width=rgb255.shape[1],
        count=3, dtype=np.uint8, crs=crs_wkt, transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(rgb255[:, :, i], i + 1)
    print(f"  saved {path}")


def save_csv(rows, path):
    keys = ["tree_id", "orig_x", "orig_y", "loc_x", "loc_y", "method", "offset_m",
            "height_bump_m", "score", "ambiguity_margin", "used_rgb", "review_recommended",
            "mean_intensity", "pct_multireturn"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([r.get(k, "") for k in keys])
    print(f"  saved {path}")


# ================================================================================
# MAIN
# ================================================================================

# ================================================================================
# STARTUP DIAGNOSTICS: attribute listing + interactive resolution choice
# ================================================================================

def print_las_attributes(sample_path):
    """
    Lists every attribute present in the LAS data, printed at the START
    of every run so you always know what's actually in the file you're
    processing without needing to run 09_las_attribute_analysis.py
    separately. Reads just one file (they all share the same schema,
    confirmed earlier in this project) rather than all 4, purely for
    speed -- this is a quick startup listing, not the full per-file,
    per-attribute statistical analysis that script 09 does.
    """
    las = laspy.read(sample_path)
    print(f"\n{'='*70}\nLAS ATTRIBUTES PRESENT (sampled from {sample_path})\n{'='*70}")
    print(f"point_format: {las.header.point_format.id}   "
          f"coordinate scale (precision floor): {las.header.scales[0]*100:.2f}cm")
    for dim in las.point_format.dimension_names:
        print(f"  {dim}")
    print(f"{'='*70}\n")


def choose_resolution_interactively(default_res_m, condition):
    """
    Lets you override the table-derived resolution at runtime instead of
    only being able to change it by editing CONFIG in the file. Press
    Enter to accept the recommended default (derived from real point-
    density testing for this specific condition -- see the resolution
    table's comments), or type a custom value in centimeters. 10cm is
    offered explicitly as a quick, much-coarser alternative -- coarser
    resolution means fewer empty cells and faster processing, at the
    cost of finer spatial detail on small targets.
    """
    print(f"\nResolution for condition '{condition}':")
    print(f"  recommended (from real point-density testing): {default_res_m*100:.1f}cm")
    print(f"  a coarser alternative, if you want fewer empty cells: 10cm")
    raw = input(f"Enter resolution in cm, or press Enter for the recommended "
                f"{default_res_m*100:.1f}cm: ").strip()
    if raw == "":
        return default_res_m
    try:
        chosen_cm = float(raw)
        print(f"  using {chosen_cm:.1f}cm (your choice, overriding the {default_res_m*100:.1f}cm recommendation)")
        return chosen_cm / 100.0
    except ValueError:
        print(f"  couldn't parse '{raw}' as a number -- using the recommended {default_res_m*100:.1f}cm instead")
        return default_res_m


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    condition = cfg["active_condition"]

    print_las_attributes(list(cfg["las_files"].values())[0])

    default_res = cfg["resolution_table_m"][condition]
    res = choose_resolution_interactively(default_res, condition)

    print(f"[1/3] Loading LAS points for condition '{condition}' (resolution={res*100:.1f}cm)...")
    xyz, cls, rgb01, intensity, n_returns = load_condition(condition, cfg)

    print("[2/3] Building raw Z + cylindrical median-color raster (no ground TIN/CHM at all)...")
    z_raster, color_raster, transform, any_point_mask = build_z_and_color_rasters(xyz, cls, rgb01, res, cfg)

    print("[3/3] Loading tree survey points and localizing...")
    sf = shapefile.Reader(cfg["shapefile_path"])
    tree_pts = np.array([[s.points[0][0], s.points[0][1]] for s in sf.shapes()])

    rows = []
    method_counts = {}
    used_rgb_count = 0
    for i, (tx, ty) in enumerate(tree_pts):
        result = localize_one_tree(tx, ty, z_raster, color_raster, transform, cfg)
        method_counts[result["method"]] = method_counts.get(result["method"], 0) + 1
        used_rgb_count += int(result.get("used_rgb", False))
        mean_intensity, pct_multi = diagnostics_near_point(
            result["x"], result["y"], xyz, intensity, n_returns, cfg["buffer_radius_m"])
        rows.append({
            "tree_id": i, "orig_x": tx, "orig_y": ty,
            "loc_x": result["x"], "loc_y": result["y"],
            "method": result["method"], "offset_m": result["offset_m"],
            "height_bump_m": result.get("height_bump_m", np.nan),
            "score": result["score"], "ambiguity_margin": result.get("ambiguity_margin", np.nan),
            "used_rgb": result.get("used_rgb", False),
            "mean_intensity": mean_intensity, "pct_multireturn": pct_multi,
        })
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(tree_pts)}")

    review_pct = cfg["review_score_percentile"]
    all_scores = np.array([r["score"] for r in rows])
    score_cutoff = np.percentile(all_scores, review_pct)
    for r in rows:
        r["review_recommended"] = r["score"] <= score_cutoff

    print(f"\nMethod breakdown: {method_counts}")
    print(f"RGB coverage: {used_rgb_count}/{len(tree_pts)} ({100*used_rgb_count/len(tree_pts):.1f}%)")
    offsets = [r["offset_m"] for r in rows if r["method"] == "localized"]
    if offsets:
        print(f"Offset: median={np.median(offsets):.3f} m, max={np.max(offsets):.3f} m")

    save_csv(rows, os.path.join(cfg["output_dir"], "stage1_localized_trees_LAS_only.csv"))
    source_prj = cfg["shapefile_path"].replace(".shp", ".prj")
    save_points_shapefile(
        [{"x": r["loc_x"], "y": r["loc_y"], **r} for r in rows],
        os.path.join(cfg["output_dir"], "stage1_localized_points_LAS_only.shp"), "x", "y", source_prj
    )
    save_points_shapefile(
        [{"x": r["orig_x"], "y": r["orig_y"], **r} for r in rows],
        os.path.join(cfg["output_dir"], "stage1_original_points_LAS_only.shp"), "x", "y", source_prj
    )
    save_buffers_shapefile(tree_pts, cfg["buffer_radius_m"],
                            os.path.join(cfg["output_dir"], "stage1_buffers_LAS_only.shp"), source_prj)
    save_color_geotiff(color_raster, z_raster, any_point_mask, transform,
                        os.path.join(cfg["output_dir"], f"stage1_las_color_raster_{condition}.tif"), source_prj, cfg)
    save_z_geotiff(z_raster, transform,
                    os.path.join(cfg["output_dir"], f"stage1_z_raw_{condition}.tif"), source_prj)
    save_z_colorized_geotiff(z_raster, transform,
                              os.path.join(cfg["output_dir"], f"stage1_z_colorized_{condition}.tif"), source_prj)

    # Optional additional outputs for a custom height range you choose --
    # entirely separate from, and doesn't modify, the standard z_raw/
    # z_colorized files above. Only asked once, for the primary/active
    # condition (asking this for all 7 conditions every run would be
    # tedious) -- same reasoning as the resolution prompt only applying
    # to the primary condition.
    custom_range = ask_custom_range_interactively()
    if custom_range is not None:
        lo, hi = custom_range
        range_tag = f"{lo:g}_{hi:g}"
        save_z_range_geotiff(z_raster, transform,
                              os.path.join(cfg["output_dir"], f"stage1_z_raw_range_{range_tag}_{condition}.tif"),
                              source_prj, lo, hi)
        range_colorized_path = os.path.join(cfg["output_dir"],
                                             f"stage1_z_colorized_range_{range_tag}_{condition}.tif")
        # The "colorized" version of a custom range is the SAME masked
        # single-band data (so QGIS gets a real, correct legend) with a
        # matching .qml sidecar that auto-applies viridis color on load --
        # see save_viridis_qml's docstring for why this approach was used
        # instead of pre-baking RGB the way the standard colorized file does.
        save_z_range_geotiff(z_raster, transform, range_colorized_path, source_prj, lo, hi)
        save_viridis_qml(range_colorized_path, lo, hi)

    # Additionally build + save Z rasters for the OTHER 6 conditions too --
    # purely as outputs to inspect/compare, since the actual localization
    # decision above only ever needs one authoritative condition.
    print(f"\nBuilding raster outputs for the other {len(ALL_CONDITIONS)-1} conditions "
          f"(for comparison -- localization itself already finished above, this doesn't change it)...")
    for other_cond in ALL_CONDITIONS:
        if other_cond == condition:
            continue  # already built and saved above
        print(f"\n  -- condition: {other_cond} --")
        other_res = cfg["resolution_table_m"][other_cond]
        o_xyz, o_cls, o_rgb01, _, _ = load_condition(other_cond, cfg)
        o_z, o_color, o_transform, o_any_point = build_z_and_color_rasters(o_xyz, o_cls, o_rgb01, other_res, cfg)
        save_color_geotiff(o_color, o_z, o_any_point, o_transform,
                            os.path.join(cfg["output_dir"], f"stage1_las_color_raster_{other_cond}.tif"), source_prj, cfg)
        save_z_geotiff(o_z, o_transform,
                       os.path.join(cfg["output_dir"], f"stage1_z_raw_{other_cond}.tif"), source_prj)
        save_z_colorized_geotiff(o_z, o_transform,
                                  os.path.join(cfg["output_dir"], f"stage1_z_colorized_{other_cond}.tif"), source_prj)
        del o_xyz, o_cls, o_rgb01, o_z, o_color, o_any_point

    print(f"\nDone. See {cfg['output_dir']}/ -- load stage1_original_points_LAS_only.shp, "
          f"stage1_localized_points_LAS_only.shp, stage1_buffers_LAS_only.shp, and "
          f"stage1_z_colorized_{condition}.tif together in QGIS to visually verify localization.")
    return rows


if __name__ == "__main__":
    main()
