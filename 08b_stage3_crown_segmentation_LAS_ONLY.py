"""
08b_stage3_crown_segmentation_LAS_ONLY.py
================================================================================
STAGE 3, REDESIGNED: LAS-only, carrying forward every Stage 1/2 lesson
================================================================================

WHAT CARRIES OVER FROM STAGE 1/2's REDESIGN, AND WHY
---------------------------------------------------------
1. NO TIF FILES. Seeds come from Stage 1's LAS-only localized positions;
   the CHM comes purely from LAS point data (Z + classification).

2. MEMORY-SAFE CHM BUILDING -- applied here from the START, not
   discovered by crashing first. Stage 2's development hit TWO separate
   real OOM crashes (confirmed via dmesg both times) building the
   all_4_merged CHM: first from concatenating full point arrays across
   sources, second from building a Delaunay triangulation from ~12
   million un-thinned ground points. Both fixes are already baked in
   here: an incremental per-source accumulation (never concatenates full
   point clouds) and ground-point thinning to the CHM's own resolution
   before triangulating.

3. REUSES STAGE 2's CHM IF IT ALREADY EXISTS, otherwise builds it fresh.
   Since Stage 3 doesn't strictly need Stage 2 to have run (this script
   is independent), it checks for
   outputs/stage2_las_only/chm_all_4_merged.tif first -- if you already
   ran Stage 2, this saves several minutes of redundant computation. If
   not, it builds the same CHM itself from scratch, using the identical
   resolution (2cm) and method, so results are consistent either way.

4. SAME per-condition resolution table as Stage 1/2 -- defaults to
   all_4_merged (2cm), the condition established as best-completeness
   throughout this project, but any of the 7 conditions can be selected.

EVERYTHING ELSE IS THE SAME PROVEN ALGORITHM AS THE ORIGINAL STAGE 3:
    multi-seed watershed (all trees flooded simultaneously so crowns
    can't overlap), hard max-crown-radius cap, marker relocation if the
    exact seed pixel is invalid, honesty about unresolvable crowns
    (too few pixels for a real 2D shape), and shape-quality filtering
    (eccentricity/circularity) to catch grass-blade-like artifacts even
    among "resolvable" crowns.
================================================================================
"""

import os
import csv
import numpy as np
import laspy
import rasterio
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from skimage.segmentation import watershed
from skimage.measure import regionprops, find_contours
from rasterio.transform import from_origin
import shapefile

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
    "active_condition": "all_4_merged",
    "resolution_table_m": {
        "60m_flight_1": 0.06, "60m_flight_2": 0.06,
        "30m_flight_1": 0.04, "30m_flight_2": 0.04,
        "60m_merged": 0.05, "30m_merged": 0.03,
        "all_4_merged": 0.02,
    },
    # Checked first -- if this file already exists (from having run
    # Stage 2), it's loaded directly instead of rebuilding from scratch.
    "reuse_chm_from": "outputs/stage2_las_only/chm_all_4_merged.tif",

    "output_dir": "outputs/stage3_las_only",

    "max_crown_radius_m": 0.5,
    "min_crown_height_m": 0.03,
    "marker_search_radius_m": 0.05,
    "min_resolvable_pixels": 15,
    "max_eccentricity": 0.9,
}


# ================================================================================
# MEMORY-SAFE CHM BUILDING (identical approach to Stage 2's fixed version)
# ================================================================================

def voxel_downsample_2d(xyz, voxel_size):
    if not voxel_size or voxel_size <= 0:
        return xyz
    voxel_idx = np.floor(xyz[:, :2] / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    return xyz[unique_idx]


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


def build_chm_incremental(xyz_list, cls_list, res, ground_thin_voxel_m=None):
    """Same memory-safe approach proven in Stage 2: never concatenates
    full point arrays across sources, and thins ground points before
    building the TIN (both fixes were needed to avoid real OOM crashes)."""
    ground_xyz = np.vstack([xyz[cls == 2] for xyz, cls in zip(xyz_list, cls_list)])
    n_before = len(ground_xyz)
    thin_voxel = ground_thin_voxel_m if ground_thin_voxel_m else res
    ground_xyz = voxel_downsample_2d(ground_xyz, thin_voxel)
    print(f"    ground points: {n_before:,} -> {len(ground_xyz):,} after thinning to {thin_voxel*100:.0f}cm")
    ground_interp = build_tin_interpolator(ground_xyz)

    xmin = min(xyz[:, 0].min() for xyz in xyz_list)
    xmax = max(xyz[:, 0].max() for xyz in xyz_list)
    ymin = min(xyz[:, 1].min() for xyz in xyz_list)
    ymax = max(xyz[:, 1].max() for xyz in xyz_list)
    ncols = int(np.ceil((xmax - xmin) / res)) + 1
    nrows = int(np.ceil((ymax - ymin) / res)) + 1
    transform = from_origin(xmin, ymax, res, res)

    flat_chm = np.full(nrows * ncols, -np.inf)
    for xyz in xyz_list:
        z_norm = xyz[:, 2] - ground_interp(xyz[:, :2])
        valid = z_norm >= -0.05
        col = np.clip(((xyz[:, 0] - xmin) / res).astype(int), 0, ncols - 1)
        row = np.clip(((ymax - xyz[:, 1]) / res).astype(int), 0, nrows - 1)
        flat_idx = row * ncols + col
        np.maximum.at(flat_chm, flat_idx[valid], z_norm[valid])
        del z_norm, valid, col, row, flat_idx

    flat_chm[flat_chm == -np.inf] = np.nan
    chm = flat_chm.reshape(nrows, ncols)
    print(f"    CHM ({res*100:.0f}cm): shape={chm.shape}, {100*np.isnan(chm).mean():.1f}% empty cells")
    return chm, transform


def load_condition_xyz_cls(condition, cfg):
    file_map = cfg["las_files"]
    if condition in file_map:
        keys = [condition]
    elif condition == "60m_merged":
        keys = ["60m_flight_1", "60m_flight_2"]
    elif condition == "30m_merged":
        keys = ["30m_flight_1", "30m_flight_2"]
    elif condition == "all_4_merged":
        keys = list(file_map.keys())
    else:
        raise ValueError(condition)
    xyz_list, cls_list = [], []
    for k in keys:
        las = laspy.read(file_map[k])
        xyz_list.append(np.column_stack([las.x, las.y, las.z]))
        cls_list.append(np.asarray(las.classification))
        print(f"    loaded {file_map[k]}: {len(las.points):,} points")
    return xyz_list, cls_list


def load_chm_geotiff(path):
    with rasterio.open(path) as src:
        chm = src.read(1)
        chm = np.where(chm == src.nodata, np.nan, chm)
        transform = src.transform
    print(f"    reused existing CHM from {path}: shape={chm.shape}")
    return chm, transform


def get_or_build_chm(cfg):
    reuse_path = cfg.get("reuse_chm_from")
    if reuse_path and os.path.exists(reuse_path):
        return load_chm_geotiff(reuse_path)
    print(f"    no existing CHM found at '{reuse_path}' -- building fresh from LAS files")
    condition = cfg["active_condition"]
    res = cfg["resolution_table_m"][condition]
    xyz_list, cls_list = load_condition_xyz_cls(condition, cfg)
    return build_chm_incremental(xyz_list, cls_list, res)


# ================================================================================
# MULTI-SEED WATERSHED SEGMENTATION (unchanged proven algorithm)
# ================================================================================

def segment_all_crowns(chm, transform, trees, cfg):
    res = transform.a
    xmin_w, ymax_w = transform.c, transform.f
    max_radius_px = cfg["max_crown_radius_m"] / res
    search_px = max(1, int(cfg.get("marker_search_radius_m", 0.05) / res))

    markers = np.zeros(chm.shape, dtype=np.int32)
    n_relocated, n_failed = 0, 0
    for i, t in enumerate(trees):
        col = int((t["x"] - xmin_w) / res)
        row = int((ymax_w - t["y"]) / res)
        if not (0 <= row < chm.shape[0] and 0 <= col < chm.shape[1]):
            continue
        val = chm[row, col]
        if np.isnan(val) or val < cfg["min_crown_height_m"]:
            r0, r1 = max(0, row - search_px), min(chm.shape[0], row + search_px + 1)
            c0, c1 = max(0, col - search_px), min(chm.shape[1], col + search_px + 1)
            local = chm[r0:r1, c0:c1]
            if np.all(np.isnan(local)) or np.nanmax(local) < cfg["min_crown_height_m"]:
                n_failed += 1
                continue
            local_r, local_c = np.unravel_index(np.nanargmax(local), local.shape)
            row, col = r0 + local_r, c0 + local_c
            n_relocated += 1
        markers[row, col] = i + 1

    print(f"    marker placement: {n_relocated} relocated, {n_failed} had no usable data nearby")

    valid_mask = ~np.isnan(chm) & (chm >= cfg["min_crown_height_m"])
    elevation = -np.nan_to_num(chm, nan=-9999)
    labels = watershed(elevation, markers=markers, mask=valid_mask)

    rr, cc = np.indices(chm.shape)
    for i, t in enumerate(trees):
        label = i + 1
        col = (t["x"] - xmin_w) / res
        row = (ymax_w - t["y"]) / res
        this_mask = labels == label
        if not this_mask.any():
            continue
        dist_px = np.hypot(rr - row, cc - col)
        too_far = this_mask & (dist_px > max_radius_px)
        labels[too_far] = 0

    return labels


def crown_stats(labels, chm, transform, trees, cfg):
    res = transform.a
    props = regionprops(labels)
    props_by_label = {p.label: p for p in props}

    results = []
    for i, t in enumerate(trees):
        label = i + 1
        mask = labels == label
        n_px = int(mask.sum())
        area_m2 = n_px * res * res
        diameter_m = 2 * np.sqrt(area_m2 / np.pi) if area_m2 > 0 else 0.0
        max_h = float(np.nanmax(chm[mask])) if n_px > 0 else np.nan
        resolvable = n_px >= cfg.get("min_resolvable_pixels", 15)

        eccentricity, circularity = np.nan, np.nan
        if label in props_by_label and resolvable:
            p = props_by_label[label]
            eccentricity = float(p.eccentricity)
            perimeter = p.perimeter if p.perimeter > 0 else np.nan
            circularity = float(4 * np.pi * p.area / (perimeter ** 2)) if not np.isnan(perimeter) else np.nan
        shape_ok = resolvable and (not np.isnan(eccentricity)) and eccentricity < cfg.get("max_eccentricity", 0.9)

        results.append({
            "tree_id": t["tree_id"], "x": t["x"], "y": t["y"],
            "crown_n_pixels": n_px, "resolvable": resolvable,
            "crown_area_m2": area_m2 if resolvable else np.nan,
            "crown_diameter_m": diameter_m if resolvable else np.nan,
            "crown_max_height_m": max_h,
            "eccentricity": eccentricity, "circularity": circularity,
            "shape_ok": shape_ok,
        })
    return results


# ================================================================================
# SAVE OUTPUTS
# ================================================================================

def save_crowns_shapefile(labels, transform, results, path, source_prj_path):
    res = transform.a
    xmin_w, ymax_w = transform.c, transform.f
    w = shapefile.Writer(path, shapeType=shapefile.POLYGON)
    w.field("tree_id", "N")
    w.field("resolv", "L")
    w.field("shape_ok", "L")
    w.field("area_m2", "N", decimal=5)
    w.field("diam_m", "N", decimal=4)
    w.field("max_h_m", "N", decimal=4)
    w.field("n_px", "N")
    w.field("eccentr", "N", decimal=4)
    w.field("circular", "N", decimal=4)

    def none_if_nan(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    n_saved = 0
    for i, r in enumerate(results):
        label = i + 1
        mask = (labels == label).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        ring = [(xmin_w + c * res, ymax_w - rr * res) for rr, c in contour]
        if len(ring) < 3:
            continue
        w.poly([ring])
        w.record(r["tree_id"], bool(r["resolvable"]), bool(r["shape_ok"]),
                  none_if_nan(r["crown_area_m2"]), none_if_nan(r["crown_diameter_m"]),
                  none_if_nan(r["crown_max_height_m"]), r["crown_n_pixels"],
                  none_if_nan(r["eccentricity"]), none_if_nan(r["circularity"]))
        n_saved += 1
    w.close()
    if source_prj_path and os.path.exists(source_prj_path):
        with open(source_prj_path) as f:
            wkt = f.read()
        with open(path.replace(".shp", ".prj"), "w") as f:
            f.write(wkt)
    print(f"  saved {path} ({n_saved}/{len(results)} crowns had a valid polygon)")


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


def save_csv(results, path):
    keys = ["tree_id", "x", "y", "crown_n_pixels", "resolvable", "shape_ok",
            "crown_area_m2", "crown_diameter_m", "crown_max_height_m",
            "eccentricity", "circularity"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in results:
            writer.writerow([r.get(k, "") for k in keys])
    print(f"  saved {path}")


def load_stage1_trees(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"tree_id": int(r["tree_id"]), "x": float(r["loc_x"]), "y": float(r["loc_y"])})
    print(f"  loaded {len(rows)} tree positions from {path}")
    return rows


# ================================================================================
# MAIN
# ================================================================================

def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("[1/3] Loading Stage 1 tree positions...")
    trees = load_stage1_trees(cfg["stage1_csv"])

    print("[2/3] Getting CHM (reusing Stage 2's if available, else building fresh)...")
    chm, transform = get_or_build_chm(cfg)
    save_chm_geotiff(chm, transform, os.path.join(cfg["output_dir"], "chm_used_for_segmentation.tif"),
                      cfg["source_prj_path"])

    print("[3/3] Segmenting crowns (multi-seed watershed, all trees at once)...")
    labels = segment_all_crowns(chm, transform, trees, cfg)
    results = crown_stats(labels, chm, transform, trees, cfg)

    n_resolvable = sum(1 for r in results if r["resolvable"])
    n_shape_ok = sum(1 for r in results if r["shape_ok"])
    resolvable_diams = np.array([r["crown_diameter_m"] for r in results if r["resolvable"]])
    print(f"\n{n_resolvable}/{len(results)} trees ({100*n_resolvable/len(results):.1f}%) got a resolvable crown "
          f"(>= {cfg['min_resolvable_pixels']} connected pixels).")
    print(f"Of those, {n_shape_ok} ({100*n_shape_ok/max(n_resolvable,1):.0f}%) also pass the shape-quality check.")
    if len(resolvable_diams) > 0:
        print(f"Among resolvable crowns: median diameter={np.median(resolvable_diams):.3f} m")

    save_csv(results, os.path.join(cfg["output_dir"], "stage3_crowns_LAS_only.csv"))
    save_crowns_shapefile(labels, transform, results,
                           os.path.join(cfg["output_dir"], "stage3_crowns_LAS_only.shp"),
                           cfg["source_prj_path"])

    print(f"\nDone. See {cfg['output_dir']}/")
    return results


if __name__ == "__main__":
    main()
