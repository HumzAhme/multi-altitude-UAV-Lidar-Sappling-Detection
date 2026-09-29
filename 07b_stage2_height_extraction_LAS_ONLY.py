"""
07b_stage2_height_extraction_LAS_ONLY.py
================================================================================
STAGE 2, REDESIGNED AND EXPANDED: LAS-only, all 7 conditions
================================================================================

WHAT THIS COVERS (everything requested, built comprehensively rather than
piecemeal, since Stage 1's redesign was already thoroughly tested and
this stage mechanically follows the same proven approach):

  1. NO TIF FILES AT ALL. Every raster (CHM) is built purely from LAS
     point data -- height comes from Z, "which points are ground" comes
     from the already-provided classification field.

  2. ALL 7 CONDITIONS, not just 3: 60m_flight_1, 60m_flight_2,
     30m_flight_1, 30m_flight_2 (each flight alone), 60m_merged,
     30m_merged, and all_4_merged. Height is extracted for every one of
     the 444 trees under every single condition, so you can compare any
     pair you want, not just the 3 the original Stage 2 covered.

  2b. RGB EXTRACTION -- NEW, added per explicit request, previously
      completely absent from this script (checked directly: zero real
      color loading existed here before). median_r/g/b per tree per
      condition, using the SAME "pure and raw" treatment 06b now uses
      for color: eligibility is simply "has real captured color", no
      artificial height-based filtering layered on top. has_real_color
      flags whether any colored point existed in that tree's buffer at
      all for that condition.

  2c. INTENSITY MADE DATA-SPECIFIC -- kept (a direct test elsewhere in
      this project found it doesn't cleanly separate "log-like" from
      "plant-like" points on its own, but it's still real, free
      diagnostic information worth having), but now ALSO reported as a
      percentile rank within this actual dataset's own distribution
      (intensity_pctile_{condition}), not just a raw absolute number
      that's meaningless without external context to compare it against.

  3. PER-CONDITION RESOLUTION -- reused directly from Stage 1's redesign,
     where each value was chosen from real points-per-cell testing (see
     that script's header for the full derivation): 60m individual=6cm,
     30m individual=4cm, 60m_merged=5cm, 30m_merged=3cm,
     all_4_merged=2cm.

  4. MERGED LAS FILES ARE NOW ACTUALLY SAVED. Previously, "merged" only
     ever existed inside a script's memory during processing -- there was
     no file you could open directly in CloudCompare to verify it
     yourself. Investigated your reported discrepancy (30m-alone showing
     points in areas the "merged" view seemed to be missing) directly:
     confirmed via a grid-cell coverage check that the raw point-level
     merge is provably complete (every cell with data in 30m-alone also
     has data in the all-4-merged set, with the merged set actually
     covering a few MORE cells) -- so that specific bug is NOT in the
     point merging itself. Most likely explanation: each condition fits
     its own separate ground surface (TIN) from a slightly different set
     of ground points, and small real differences in that fitted surface
     can shift which specific cells clear the height threshold -- a real,
     subtle effect, not literally "missing merged data". Saving the
     actual merged .las files now (see OUTPUTS below) so you can check
     this claim directly yourself in CloudCompare, rather than trust it
     from a script's console output.

  5. INTENSITY AND MULTI-RETURN as per-tree DIAGNOSTIC columns (not
     driving the height calculation) -- tested separately in Stage 1's
     development and found NOT to cleanly separate "log-like" from
     "plant-like" points on this data, but still recorded here per-tree
     per-condition so you can inspect them yourself.

HOW TO SEE COLOR-BY-HEIGHT INSTEAD OF GRAY (read this before assuming
anything is broken)
--------------------------------------------------------------------------
This is a VIEWER SETTING in both CloudCompare and QGIS, not something
wrong with the files:

  IN CLOUDCOMPARE (for the merged .las point cloud files below):
    1. Load the .las file (accept the Global Shift prompt if it appears).
    2. In the DB Tree (left panel), click the cloud to select it.
    3. In the Properties panel (bottom left), find "Colors" and change it
       from RGB to a scalar field -- if height-above-ground isn't already
       a scalar field, you can compute one: Edit > Scalar Fields >
       ... or simply display by the raw "Z" coordinate as a quick proxy
       for height (not normalized above ground, but often visually similar
       for a single flat plot like this one).
    4. Once a height-like scalar field is active, use the color ramp
       icon (usually a small rainbow icon in the toolbar, or
       Edit > Scalar fields > Show SF colors scale) to pick a colorful
       (e.g. "Blue to Red" or "Jet") ramp instead of grayscale.
    5. If it's still gray: check whether "Colors" is set to display RGB
       (real color) rather than the scalar field -- RGB display looks
       "grayish/dull" for this data because outdoor natural scenes are
       genuinely mostly brown/green/gray, which is real color, not a
       display bug. Switch specifically to the SCALAR FIELD display mode
       to get an artificial height-based rainbow color instead.

  IN QGIS (for the chm_*.tif GeoTIFF files below):
    1. Add the raster layer (Layer > Add Layer > Add Raster Layer).
    2. By default, a single-band float raster like a CHM renders in
       plain grayscale -- this is QGIS's default behavior for ANY
       single-band raster, not specific to this file or a bug.
    3. Right-click the layer > Properties > Symbology.
    4. Change "Render type" from "Singleband gray" to
       "Singleband pseudocolor".
    5. Pick a color ramp (e.g. "Spectral" or "Viridis" -- Viridis is
       used in this project's own matplotlib comparison plots, so
       picking it here keeps the visual language consistent between
       QGIS and the PNGs this pipeline already generates).
    6. Click "Classify" to auto-generate a color scale from the actual
       min/max height values in the file, then Apply.

  TIP -- SAPLING HEIGHT RANGE: these seedlings are mostly 0-0.6m tall
  (confirmed range from real data). When setting the color ramp's min/max
  in either tool, using the DATA's actual full range (which can include
  taller real trees/noise up to ~2m) will squash most of your small
  seedlings into a narrow, hard-to-distinguish color band. Manually
  setting the ramp's max to something like 0.5-0.8m instead of the full
  data range will spread the color variation across the height range
  that actually matters for your saplings -- worth doing manually rather
  than relying on "auto-classify using full data range" for this
  specific plot.
================================================================================
"""

import os
import csv
import numpy as np
import laspy
import rasterio
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    "stage1_csv": "outputs/stage1_las_only/stage1_localized_trees_LAS_only.csv",
    "source_prj_path": "trees_crop.prj",

    "las_files": {
        "60m_flight_1": "60m/260522_122413_filt_crop.las",
        "60m_flight_2": "60m/260522_124721_filt_crop.las",
        "30m_flight_1": "30m/260522_130956_filt_crop.las",
        "30m_flight_2": "30m/260522_140126_filt_crop.las",
    },

    "output_dir": "outputs/stage2_las_only",

    # SAME per-condition resolution table as Stage 1's redesign, derived
    # from real points-per-cell testing (see that script's header).
    "resolution_table_m": {
        "60m_flight_1": 0.06, "60m_flight_2": 0.06,
        "30m_flight_1": 0.04, "30m_flight_2": 0.04,
        "60m_merged": 0.05, "30m_merged": 0.03,
        "all_4_merged": 0.02,
    },

    # Tight buffer for height EXTRACTION specifically (Stage 1 already
    # refined the position with its own, separate, wider search buffer --
    # this one only needs to be robust to a pixel or two of residual
    # noise around that already-refined point).
    "height_buffer_radius_m": 0.12,
    "min_points_for_confidence": 5,

    # Which merged conditions to actually write out as real .las files
    # you can open directly in CloudCompare (see header comment).
    "save_merged_las": ["60m_merged", "30m_merged", "all_4_merged"],
}

ALL_CONDITIONS = ["60m_flight_1", "60m_flight_2", "30m_flight_1", "30m_flight_2",
                   "60m_merged", "30m_merged", "all_4_merged"]


# ================================================================================
# LOAD EACH RAW FILE ONCE, BUILD ALL 7 CONDITIONS FROM THOSE (no redundant disk reads)
# ================================================================================

def load_all_raw_files(cfg):
    """Reads each of the 4 physical files exactly once; returns a dict of
    (las_object, xyz, cls, intensity, n_returns, rgb01) keyed by flight
    label. rgb01 added here -- previously Stage 2 never loaded color at
    all, despite it being real, available data in every LAS file (same
    point format 7 confirmed throughout this project). Scaled to [0,1]
    the same way 06b does it (LAS RGB can be stored as either 8-bit or
    16-bit depending on the source -- checking which one this data uses
    rather than assuming, so the scaling is always correct regardless)."""
    raw = {}
    for label, path in cfg["las_files"].items():
        las = laspy.read(path)
        xyz = np.column_stack([las.x, las.y, las.z])
        cls = np.asarray(las.classification)
        intensity = np.asarray(las.intensity, dtype=float)
        n_returns = np.asarray(las.number_of_returns)
        r, g, b = np.asarray(las.red, dtype=float), np.asarray(las.green, dtype=float), np.asarray(las.blue, dtype=float)
        scale = 65535.0 if max(r.max(), g.max(), b.max()) > 255 else 255.0
        rgb01 = np.stack([r, g, b], axis=1) / scale
        raw[label] = {"las": las, "xyz": xyz, "cls": cls,
                      "intensity": intensity, "n_returns": n_returns, "rgb01": rgb01}
        print(f"    loaded {path}: {len(xyz):,} points")
    return raw


def get_condition_source_keys(condition, raw):
    """Returns which raw file keys make up a given condition, without
    combining their arrays -- the caller processes each source separately."""
    if condition in raw:
        return [condition]
    if condition == "60m_merged":
        return ["60m_flight_1", "60m_flight_2"]
    if condition == "30m_merged":
        return ["30m_flight_1", "30m_flight_2"]
    if condition == "all_4_merged":
        return list(raw.keys())
    raise ValueError(condition)


def save_merged_las(condition, raw, out_path):
    """Actually writes a merged .las file to disk -- this is the file you
    can load directly into CloudCompare to verify the merge yourself."""
    if condition == "60m_merged":
        keys = ["60m_flight_1", "60m_flight_2"]
    elif condition == "30m_merged":
        keys = ["30m_flight_1", "30m_flight_2"]
    elif condition == "all_4_merged":
        keys = list(raw.keys())
    else:
        raise ValueError(condition)
    clouds = [raw[k]["las"] for k in keys]
    merged = laspy.LasData(clouds[0].header)
    merged.points = laspy.ScaleAwarePointRecord(
        np.concatenate([c.points.array for c in clouds]),
        clouds[0].point_format, clouds[0].header.scales, clouds[0].header.offsets
    )
    merged.write(out_path)
    print(f"  saved {out_path}  ({len(merged.points):,} points -- open this directly "
          f"in CloudCompare to verify the merge yourself)")


# ================================================================================
# GROUND / CHM (same TIN approach used throughout this project)
# ================================================================================

def build_tin_interpolator(ground_xyz):
    lin = LinearNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])
    near = NearestNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])

    def interp(xy):
        z = lin(xy)
        nan_mask = np.isnan(z)
        if nan_mask.any():
            z[nan_mask] = near(xy[nan_mask])
        return z
    return interp


def voxel_downsample_2d(xyz, voxel_size):
    """
    Thins a point cloud to at most one point per voxel_size x,y grid
    cell -- same proven technique used earlier in this project. A
    Delaunay triangulation (which LinearNDInterpolator builds internally)
    gets dramatically more expensive as point count grows, but gains
    nothing from ground points denser than the CHM's own output
    resolution anyway. CONFIRMED NECESSARY BY AN ACTUAL SECOND CRASH:
    even after removing the full-point-cloud concatenation, building the
    TIN from all_4_merged's full ~10+ million ground points alone was
    still enough to OOM-kill the process a second time -- this thinning
    is what actually fixes it, not a preventative guess.
    """
    if not voxel_size or voxel_size <= 0:
        return xyz
    voxel_idx = np.floor(xyz[:, :2] / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    return xyz[unique_idx]


def build_chm_incremental(xyz_list, cls_list, res, ground_thin_voxel_m=None):
    """
    Builds the CHM/count rasters by processing each SOURCE file's points
    one at a time and accumulating into shared output rasters -- never
    concatenates the full point arrays into one big copy.

    WHY THIS MATTERS -- CONFIRMED BY TWO ACTUAL CRASHES, NOT A HYPOTHETICAL:
    the original version of this function took one already-concatenated
    xyz array, which roughly doubled peak memory for "all_4_merged"
    (confirmed via dmesg: OOM-killed at 3.94GB). Fixing that (this
    incremental version) helped but was NOT enough on its own -- a SECOND
    crash (also confirmed via dmesg, 3.95GB) traced to building the
    ground TIN's Delaunay triangulation from the full, un-thinned ground
    point set (~10+ million points for this condition). Ground points are
    now thinned to `ground_thin_voxel_m` (defaults to the CHM resolution
    itself) before triangulating -- the DTM is only ever evaluated at
    that resolution anyway, so this loses no real accuracy.
    """
    ground_xyz = np.vstack([xyz[cls == 2] for xyz, cls in zip(xyz_list, cls_list)])
    n_ground_before = len(ground_xyz)
    thin_voxel = ground_thin_voxel_m if ground_thin_voxel_m else res
    ground_xyz = voxel_downsample_2d(ground_xyz, thin_voxel)
    print(f"    ground points: {n_ground_before:,} -> {len(ground_xyz):,} after thinning "
          f"to {thin_voxel*100:.0f}cm for TIN construction")
    ground_interp = build_tin_interpolator(ground_xyz)

    xmin = min(xyz[:, 0].min() for xyz in xyz_list)
    xmax = max(xyz[:, 0].max() for xyz in xyz_list)
    ymin = min(xyz[:, 1].min() for xyz in xyz_list)
    ymax = max(xyz[:, 1].max() for xyz in xyz_list)
    ncols = int(np.ceil((xmax - xmin) / res)) + 1
    nrows = int(np.ceil((ymax - ymin) / res)) + 1
    transform = from_origin(xmin, ymax, res, res)

    flat_chm = np.full(nrows * ncols, -np.inf)
    flat_count = np.zeros(nrows * ncols, dtype=int)

    for xyz in xyz_list:
        z_norm = xyz[:, 2] - ground_interp(xyz[:, :2])
        valid = z_norm >= -0.05
        col = np.clip(((xyz[:, 0] - xmin) / res).astype(int), 0, ncols - 1)
        row = np.clip(((ymax - xyz[:, 1]) / res).astype(int), 0, nrows - 1)
        flat_idx = row * ncols + col
        np.maximum.at(flat_chm, flat_idx[valid], z_norm[valid])
        np.add.at(flat_count, flat_idx[valid], 1)
        del z_norm, valid, col, row, flat_idx  # explicit, since these loop-local arrays

    flat_chm[flat_chm == -np.inf] = np.nan
    chm = flat_chm.reshape(nrows, ncols)
    count_raster = flat_count.reshape(nrows, ncols)
    print(f"    CHM ({res*100:.0f}cm): shape={chm.shape}, {100*np.isnan(chm).mean():.1f}% empty cells")
    return chm, count_raster, transform, ground_interp


def save_chm_geotiff(chm, transform, path, source_prj_path):
    crs_wkt = None
    if source_prj_path and os.path.exists(source_prj_path):
        with open(source_prj_path) as f:
            crs_wkt = f.read()
    chm_out = np.where(np.isnan(chm), -9999, chm).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=chm.shape[0], width=chm.shape[1],
                        count=1, dtype=np.float32, nodata=-9999, crs=crs_wkt, transform=transform) as dst:
        dst.write(chm_out, 1)
    print(f"  saved {path}")


# ================================================================================
# PER-TREE EXTRACTION: height + point count + intensity/multi-return diagnostics
# ================================================================================

def extract_for_tree(chm, count_raster, transform, cx, cy, radius_m, source_list):
    """
    source_list: list of dicts, each with keys xyz/cls/intensity/n_returns/
    rgb01 for ONE raw file -- accumulated here across however many sources
    make up this condition, without ever concatenating them into one array
    (same memory-safe pattern proven necessary elsewhere in this project).

    RGB EXTRACTION -- NEW, added per explicit request, matching how 06b
    treats color: PURE and RAW, no artificial height-based filtering.
    Reports the MEDIAN captured color (not mean -- more robust to one
    stray outlier-colored point, same reasoning as 06b's color raster)
    of every REAL colored point in this tree's buffer, from EVERY source
    file that contributes to this condition. Eligibility is simply "has
    real color" -- exactly matching 06b's current color_eligible
    definition, not the old (now-removed) height-above-ground
    requirement, so the two scripts treat color consistently with each
    other.

    INTENSITY -- kept, but see the note in run_significance_tests /
    main() for how this gets turned into something more genuinely
    data-specific (a percentile rank within THIS dataset) rather than
    left as a raw absolute number that's hard to interpret without
    external context. Confirmed relevant to keep: even though intensity
    didn't cleanly separate "log-like" from "plant-like" points in an
    earlier direct test (see script 09's findings), it's still real,
    free-to-compute diagnostic information worth having available, not
    something to discard just because it didn't turn out to be a
    strong standalone signal for that one specific purpose.
    """
    res = transform.a
    xmin_w, ymax_w = transform.c, transform.f
    col_c = (cx - xmin_w) / res
    row_c = (ymax_w - cy) / res
    half_px = int(np.ceil(radius_m / res))

    r0, r1 = max(0, int(row_c - half_px)), min(chm.shape[0], int(row_c + half_px) + 1)
    c0, c1 = max(0, int(col_c - half_px)), min(chm.shape[1], int(col_c + half_px) + 1)
    if r1 <= r0 or c1 <= c0:
        return dict(height=np.nan, n_points=0, mean_intensity=np.nan, pct_multireturn=np.nan,
                    median_r=np.nan, median_g=np.nan, median_b=np.nan, has_real_color=False)

    chm_win = chm[r0:r1, c0:c1]
    count_win = count_raster[r0:r1, c0:c1]
    rr, cc = np.indices(chm_win.shape)
    dist_px = np.hypot(rr - (row_c - r0), cc - (col_c - c0))
    inside = dist_px <= (radius_m / res)

    n_points = int(count_win[inside].sum())
    valid_heights = chm_win[inside & ~np.isnan(chm_win)]
    height = float(valid_heights.max()) if len(valid_heights) > 0 else np.nan

    # Accumulate intensity/multi-return/color across each source
    # separately -- these per-source arrays are far smaller than a full
    # merged copy would be, and nothing here is ever concatenated across
    # sources (same memory-safe approach used throughout this project).
    intensity_sum, intensity_n, multi_n, total_n = 0.0, 0, 0, 0
    all_colored_rgb = []  # small per-tree list, not a whole-cloud copy -- cheap
    for src in source_list:
        xyz = src["xyz"]
        dxy = np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy)
        in_buffer = dxy <= radius_m
        if in_buffer.any():
            intensity_sum += float(src["intensity"][in_buffer].sum())
            intensity_n += int(in_buffer.sum())
            multi_n += int((src["n_returns"][in_buffer] > 1).sum())
            total_n += int(in_buffer.sum())
            buf_rgb = src["rgb01"][in_buffer]
            has_color = buf_rgb.sum(axis=1) > 0  # pure raw check -- same as 06b, no height filter
            if has_color.any():
                all_colored_rgb.append(buf_rgb[has_color])
        del dxy, in_buffer

    mean_intensity = intensity_sum / intensity_n if intensity_n > 0 else np.nan
    pct_multi = 100 * multi_n / total_n if total_n > 0 else np.nan

    if all_colored_rgb:
        stacked = np.vstack(all_colored_rgb)
        median_rgb = np.median(stacked, axis=0)  # median, not mean -- robust to one outlier point
        median_r, median_g, median_b = float(median_rgb[0]), float(median_rgb[1]), float(median_rgb[2])
        has_real_color = True
    else:
        median_r, median_g, median_b = np.nan, np.nan, np.nan
        has_real_color = False

    return dict(height=height, n_points=n_points, mean_intensity=mean_intensity, pct_multireturn=pct_multi,
                median_r=median_r, median_g=median_g, median_b=median_b, has_real_color=has_real_color)



# ================================================================================
# MAIN
# ================================================================================

def load_stage1_trees(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"tree_id": int(r["tree_id"]), "x": float(r["loc_x"]), "y": float(r["loc_y"])})
    print(f"  loaded {len(rows)} tree positions from {path}")
    return rows


# ================================================================================
# COMPARISON PLOTS AND SIGNIFICANCE TESTS
# ================================================================================
# RESTORED HERE -- these existed in the original (pre-LAS-only) Stage 2
# script but were dropped entirely during the LAS-only rewrite (confirmed
# directly: zero plotting or statistics code existed in this file until
# now). With 7 conditions instead of 3, showing all 21 possible pairs
# would be unreadable -- instead this focuses on the 5 comparisons that
# actually answer this project's real questions: within-altitude
# consistency (are the two 60m flights consistent with each other? the
# two 30m flights?), the core altitude question (60m-merged vs
# 30m-merged), and whether adding the other altitude's data to a merge
# changes its answer (30m-merged vs all-4-merged, 60m-merged vs
# all-4-merged).

MEANINGFUL_PAIRS = [
    ("60m_flight_1", "60m_flight_2", "60m flights: internally consistent?"),
    ("30m_flight_1", "30m_flight_2", "30m flights: internally consistent?"),
    ("60m_merged", "30m_merged", "the core altitude question"),
    ("30m_merged", "all_4_merged", "does adding 60m change the 30m answer?"),
    ("60m_merged", "all_4_merged", "does adding 30m change the 60m answer?"),
]


def make_comparison_plots(out_rows, out_dir):
    """Scatter plot per meaningful pair (5 total) against the 1:1 line --
    real seedlings should cluster near that line if two conditions agree;
    a systematic offset (not just scatter) indicates a real, consistent
    bias, the same interpretation used throughout this project."""
    fig, axes = plt.subplots(1, len(MEANINGFUL_PAIRS), figsize=(5 * len(MEANINGFUL_PAIRS), 5))
    for ax, (cond_a, cond_b, label) in zip(axes, MEANINGFUL_PAIRS):
        a = np.array([r[f"height_{cond_a}"] for r in out_rows], dtype=float)
        b = np.array([r[f"height_{cond_b}"] for r in out_rows], dtype=float)
        valid = ~(np.isnan(a) | np.isnan(b))
        ax.scatter(a[valid], b[valid], s=8, alpha=0.4)
        lim = [0, max(np.nanmax(a), np.nanmax(b)) * 1.05]
        ax.plot(lim, lim, "r--", linewidth=1, label="1:1 line")
        ax.set_xlabel(f"{cond_a} height (m)")
        ax.set_ylabel(f"{cond_b} height (m)")
        ax.set_title(f"{label}\n(n={valid.sum()})", fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "height_comparison_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def make_distribution_plot(out_rows, out_dir):
    """All 7 conditions' full height distributions overlaid -- shows the
    whole-dataset shape (e.g. 60m's leftward/lower shift), not just a
    single median number per condition."""
    fig, ax = plt.subplots(figsize=(9, 6))
    all_vals = np.concatenate([[r[f"height_{c}"] for r in out_rows] for c in ALL_CONDITIONS])
    all_vals = all_vals[~np.isnan(all_vals.astype(float))]
    bins = np.linspace(0, np.nanpercentile(all_vals.astype(float), 98), 40)
    for cond in ALL_CONDITIONS:
        vals = np.array([r[f"height_{cond}"] for r in out_rows], dtype=float)
        v = vals[~np.isnan(vals)]
        ax.hist(v, bins=bins, alpha=0.35, label=f"{cond} (median={np.median(v):.3f}m)")
    ax.set_xlabel("height (m)")
    ax.set_ylabel("count")
    ax.set_title("Height distribution across all 444 trees, by condition (all 7)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "height_distribution_all_conditions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def run_significance_tests(out_rows):
    """Wilcoxon signed-rank test (appropriate for paired, non-normally-
    distributed data like these small, right-skewed heights -- a paired
    t-test's normality assumption doesn't hold well here) for each of the
    5 meaningful pairs. Answers 'is this difference likely real' with an
    actual p-value, not just eyeballing a median gap."""
    from scipy.stats import wilcoxon
    print("\n=== Wilcoxon signed-rank tests (is the difference statistically real?) ===")
    for cond_a, cond_b, label in MEANINGFUL_PAIRS:
        a = np.array([r[f"height_{cond_a}"] for r in out_rows], dtype=float)
        b = np.array([r[f"height_{cond_b}"] for r in out_rows], dtype=float)
        valid = ~(np.isnan(a) | np.isnan(b))
        d = a[valid] - b[valid]
        d_nonzero = d[d != 0]
        if len(d_nonzero) < 10:
            print(f"  {cond_a} vs {cond_b} ({label}): too few non-tied pairs to test")
            continue
        stat, p = wilcoxon(d_nonzero)
        sig = "YES -- statistically significant" if p < 0.05 else "not significant at p<0.05"
        print(f"  {cond_a} vs {cond_b} ({label}): p={p:.2e}  ({sig}, n={len(d_nonzero)} pairs, "
              f"median diff={np.median(d_nonzero):+.4f}m)")


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("[1/3] Loading Stage 1 tree positions...")
    trees = load_stage1_trees(cfg["stage1_csv"])

    print("[2/3] Loading all 4 raw LAS files once...")
    raw = load_all_raw_files(cfg)

    for cond in cfg["save_merged_las"]:
        save_merged_las(cond, raw, os.path.join(cfg["output_dir"], f"merged_{cond}.las"))

    print("[3/3] Building CHM + extracting height for all 7 conditions...")
    per_condition_results = {}
    for cond in ALL_CONDITIONS:
        print(f"\n  -- condition: {cond} --")
        res = cfg["resolution_table_m"][cond]
        source_keys = get_condition_source_keys(cond, raw)
        source_list = [raw[k] for k in source_keys]
        xyz_list = [s["xyz"] for s in source_list]
        cls_list = [s["cls"] for s in source_list]

        chm, count_raster, transform, ground_interp = build_chm_incremental(xyz_list, cls_list, res)
        save_chm_geotiff(chm, transform, os.path.join(cfg["output_dir"], f"chm_{cond}.tif"),
                          cfg["source_prj_path"])

        cond_results = []
        for t in trees:
            r = extract_for_tree(chm, count_raster, transform, t["x"], t["y"],
                                  cfg["height_buffer_radius_m"], source_list)
            cond_results.append(r)
        per_condition_results[cond] = cond_results

        heights = np.array([r["height"] for r in cond_results])
        npts = np.array([r["n_points"] for r in cond_results])
        print(f"    median height={np.nanmedian(heights):.3f}m, "
              f"zero-point trees={int((npts==0).sum())} ({100*(npts==0).mean():.1f}%), "
              f"low-confidence(<{cfg['min_points_for_confidence']}pts)="
              f"{int((npts<cfg['min_points_for_confidence']).sum())} "
              f"({100*(npts<cfg['min_points_for_confidence']).mean():.1f}%)")

    # --- write one comprehensive CSV, all 7 conditions x all diagnostics ---
    out_rows = []
    for i, t in enumerate(trees):
        row = {"tree_id": t["tree_id"], "x": t["x"], "y": t["y"]}
        for cond in ALL_CONDITIONS:
            r = per_condition_results[cond][i]
            row[f"height_{cond}"] = r["height"]
            row[f"n_points_{cond}"] = r["n_points"]
            row[f"low_confidence_{cond}"] = r["n_points"] < cfg["min_points_for_confidence"]
            row[f"mean_intensity_{cond}"] = r["mean_intensity"]
            row[f"pct_multireturn_{cond}"] = r["pct_multireturn"]
            row[f"median_r_{cond}"] = r["median_r"]
            row[f"median_g_{cond}"] = r["median_g"]
            row[f"median_b_{cond}"] = r["median_b"]
            row[f"has_real_color_{cond}"] = r["has_real_color"]
        out_rows.append(row)

    # DATA-SPECIFIC INTENSITY REFINEMENT -- per explicit request: rather
    # than leave mean_intensity as a raw absolute number (hard to judge
    # without external context -- is 42,000 high or low for THIS data?),
    # also compute each tree's PERCENTILE RANK within this actual
    # dataset's own real intensity distribution, per condition. This has
    # to happen as a separate pass AFTER every tree's raw mean_intensity
    # is known (needs the whole distribution first), not inside
    # extract_for_tree, which only ever sees one tree at a time.
    # "intensity_pctile_{cond}=87" is immediately interpretable (this
    # tree is brighter/more reflective than 87% of all trees under this
    # condition) in a way a raw absolute number never is on its own.
    for cond in ALL_CONDITIONS:
        vals = np.array([row[f"mean_intensity_{cond}"] for row in out_rows], dtype=float)
        valid = ~np.isnan(vals)
        ranks = np.full(len(vals), np.nan)
        if valid.sum() > 1:
            from scipy.stats import rankdata
            ranks[valid] = 100 * (rankdata(vals[valid]) - 1) / (valid.sum() - 1)
        for i, row in enumerate(out_rows):
            row[f"intensity_pctile_{cond}"] = round(float(ranks[i]), 1) if not np.isnan(ranks[i]) else ""

    keys = ["tree_id", "x", "y"]
    for cond in ALL_CONDITIONS:
        keys += [f"height_{cond}", f"n_points_{cond}", f"low_confidence_{cond}",
                 f"mean_intensity_{cond}", f"intensity_pctile_{cond}", f"pct_multireturn_{cond}",
                 f"median_r_{cond}", f"median_g_{cond}", f"median_b_{cond}", f"has_real_color_{cond}"]
    out_path = os.path.join(cfg["output_dir"], "stage2_heights_all_conditions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in out_rows:
            writer.writerow([row.get(k, "") for k in keys])
    print(f"\nsaved {out_path}  ({len(keys)} columns x {len(out_rows)} trees)")

    print("\nBuilding comparison plots...")
    make_comparison_plots(out_rows, cfg["output_dir"])
    make_distribution_plot(out_rows, cfg["output_dir"])
    run_significance_tests(out_rows)

    print(f"\nDone. See {cfg['output_dir']}/ -- includes real merged .las files "
          f"(open directly in CloudCompare) and per-condition CHM GeoTIFFs "
          f"(open in QGIS -- see this script's header for how to color them by height).")
    return out_rows


if __name__ == "__main__":
    main()
