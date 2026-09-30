"""
11_verify_against_drone_tif.py
================================================================================
DIAGNOSTIC: check LAS-localized positions against the REAL drone photos
================================================================================

WHAT THIS DOES AND WHY IT EXISTS
-------------------------------------
Confirmed directly: sampling the real drone .tif photos at Stage 1's
LAS-derived coordinates gives a MUCH clearer view than the LAS-only color
raster's reconstructed image -- real photographic detail vs. sparse
point-based extrapolation. This does NOT change the actual processing
pipeline at all -- localization, height, and segmentation all remain
100% LAS-only, exactly as already built. This script is purely a
verification convenience: it batches the same kind of check across many
trees at once, instead of you manually panning to each one in QGIS.

You don't strictly need this script to do the same check -- you can just
add the original .tif files as QGIS layers alongside the shapefiles
Stage 1 already produces, since both use the same real coordinate
system. This script exists for convenience: reviewing 20-40 trees at
once in a grid is faster than panning to each one individually.

ONE HONEST CAVEAT, worth remembering while reviewing the output: the
drone photos were confirmed early in this project to be only "roughly"
georeferenced (unlike the LAS point clouds, which are properly
GNSS/IMU-registered). A small, consistent-looking offset between the red
cross and a visible plant across MANY trees could reflect that rough
photo georeferencing, not a real localization error -- a LARGE or
inconsistent offset is more likely a genuine problem worth investigating.
================================================================================
"""

import os
import csv
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    "stage1_csv": "outputs/stage1_las_only/stage1_localized_trees_LAS_only.csv",
    "tif_files": [
        "30m/DSC02145_georef_1cm_crop.tif",
        "30m/DSC02146_georef_1cm_crop.tif",
        "30m/DSC02147_georef_1cm_crop.tif",
        "30m/DSC02148_georef_1cm_crop.tif",
        "30m/DSC02149_georef_1cm_crop.tif",
        "60m/DSC00411_georef_1cm_crop.tif",
    ],
    "output_dir": "outputs/tif_verification",
    "view_radius_m": 0.3,
    "buffer_radius_m": 0.2,
    # Which trees to check -- "all" for every tree, "localized_only" for
    # every non-fallback tree (still likely 400+, many grid images),
    # "sample" for a random sample of `sample_size` trees (the sensible
    # default for a quick review -- a first run with "localized_only"
    # produced 26 separate 16-tree images, more than anyone would
    # actually want to click through at once).
    "which_trees": "sample",
    "sample_size": 40,
    "max_trees_per_image": 16,
    "random_seed": 7,
}


# ================================================================================
# SAMPLE THE REAL TIF AT A GIVEN COORDINATE
# ================================================================================

def open_tif_sources(paths):
    return [rasterio.open(p) for p in paths]


def get_tif_crop(srcs, cx, cy, half_m):
    """Checks each TIF in order, returns the first one with real
    (non-black) data at this location -- same multi-photo coverage
    handling used throughout this project, since the photos don't fully
    overlap with each other."""
    for src in srcs:
        try:
            res = src.res[0]
            half_px = int(half_m / res)
            row, col = src.index(cx, cy)
            window = rasterio.windows.Window(col - half_px, row - half_px, half_px * 2 + 1, half_px * 2 + 1)
            win = src.read(window=window)
            win = np.moveaxis(win, 0, -1)
            if win.shape[0] > 0 and win.shape[1] > 0 and win[win.shape[0] // 2, win.shape[1] // 2].sum() > 0:
                return win
        except Exception:
            continue
    return None


def load_trees(csv_path, which_trees, sample_size, seed):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if which_trees == "all":
        return rows
    if which_trees == "localized_only":
        return [r for r in rows if r["method"] == "localized"]
    if which_trees == "sample":
        localized = [r for r in rows if r["method"] == "localized"]
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(localized), min(sample_size, len(localized)), replace=False)
        return [localized[i] for i in idx]
    raise ValueError(f"Unknown which_trees option: {which_trees}")


# ================================================================================
# MAIN
# ================================================================================

def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("[1/3] Reading Stage 1 output...")
    trees = load_trees(cfg["stage1_csv"], cfg["which_trees"], cfg["sample_size"], cfg["random_seed"])
    print(f"  checking {len(trees)} trees")

    print("[2/3] Opening drone TIF sources...")
    srcs = open_tif_sources(cfg["tif_files"])

    print(f"[3/3] Building comparison grids...")
    half = cfg["view_radius_m"]
    n_per_image = cfg["max_trees_per_image"]
    n_images = int(np.ceil(len(trees) / n_per_image))
    n_no_coverage = 0

    for img_i in range(n_images):
        batch = trees[img_i * n_per_image: (img_i + 1) * n_per_image]
        ncols = 4
        nrows = int(np.ceil(len(batch) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)

        for i, r in enumerate(batch):
            tid = r["tree_id"]
            tx, ty = float(r["loc_x"]), float(r["loc_y"])
            crop = get_tif_crop(srcs, tx, ty, half)
            ax = axes[i // ncols, i % ncols]
            if crop is not None:
                ax.imshow(crop.astype(np.uint8), extent=(-half, half, -half, half))
            else:
                ax.set_facecolor("black")
                ax.text(0.5, 0.5, "no TIF\ncoverage here", ha="center", va="center",
                         color="white", transform=ax.transAxes, fontsize=9)
                n_no_coverage += 1
            ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2)
            circ = plt.Circle((0, 0), cfg["buffer_radius_m"], fill=False, color="yellow", linewidth=0.8)
            ax.add_patch(circ)
            ax.set_title(f"tree {tid}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        for j in range(len(batch), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.suptitle("Red = LAS-derived localized position, on the REAL drone photo "
                      "(not the LAS-reconstructed raster)")
        plt.tight_layout()
        out_path = os.path.join(cfg["output_dir"], f"tif_verification_{img_i+1}.png")
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"  saved {out_path} ({len(batch)} trees)")

    print(f"\n{n_no_coverage}/{len(trees)} trees had no real photo coverage at all "
          f"(from any of the {len(cfg['tif_files'])} TIFs) -- these show as black panels, "
          f"same multi-photo coverage limitation established earlier in this project.")
    print(f"\nDone. See {cfg['output_dir']}/ -- remember the photos are only roughly "
          f"georeferenced (see this script's header), so judge small, consistent offsets "
          f"differently than large or inconsistent ones.")


if __name__ == "__main__":
    main()
