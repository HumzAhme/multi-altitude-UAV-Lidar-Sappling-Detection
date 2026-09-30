"""
10_stage1_fallback_diagnostic.py
================================================================================
DIAGNOSTIC: visualize every tree where Stage 1 localization fell back
================================================================================

WHAT THIS DOES AND WHY IT'S A SEPARATE SCRIPT
--------------------------------------------------
Stage 1's console output reports a count of "fallback_zero_score" and
"fallback_no_height_bump" trees, but doesn't show you WHICH trees these
are or WHAT their local area actually looks like -- you'd have to
manually cross-reference tree_ids and rebuild crops yourself. This
script does that automatically: it reads your actual Stage 1 output CSV,
finds every tree that fell back (for whichever reason), and rebuilds the
same RGB + CHM + color-anomaly diagnostic view used earlier in this
project to debug the tree 238/109 issues -- except now for every
fallback tree in YOUR real run, not just a hand-picked few.

This has to be a separate script (not something run inline) because the
specific trees that fell back are particular to your actual data and
your actual Stage 1 run -- there's no way to know which tree_ids these
are without reading your real output file.

WHAT TO DO WITH THE RESULT
-------------------------------
Look at each panel yourself, OR save/send the output image back for
review -- either way, the goal is to answer: is this tree genuinely
unlocatable from height+color alone (like the confirmed dense-grass
case from earlier), or is there a real, fixable pattern worth
investigating further?
================================================================================
"""

import os
import csv
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================================
# CONFIGURATION -- mirrors Stage 1's own settings, since this rebuilds
# the exact same view Stage 1 itself used internally
# ================================================================================

CONFIG = {
    "stage1_csv": "outputs/stage1_las_only/stage1_localized_trees_LAS_only.csv",
    "las_files": [
        "60m/260522_122413_filt_crop.las",
        "60m/260522_124721_filt_crop.las",
        "30m/260522_130956_filt_crop.las",
        "30m/260522_140126_filt_crop.las",
    ],
    "output_dir": "outputs/stage1_diagnostic",
    "view_radius_m": 0.3,          # slightly wider than the 0.2m buffer, for context
    "buffer_radius_m": 0.2,        # Stage 1's actual search radius, drawn as a circle
    "local_contrast_radius_m": 0.08,
    "max_trees_per_image": 16,     # splits into multiple grid images if more than this
    "fallback_methods": ["fallback_zero_score", "fallback_no_height_bump"],
}


# ================================================================================
# LOAD ALL POINTS ONCE (small task, no need for Stage 2/3's incremental
# machinery here -- diagnostics only look at small local crops)
# ================================================================================

def load_all_points(paths):
    all_xyz, all_rgb = [], []
    for p in paths:
        las = laspy.read(p)
        all_xyz.append(np.column_stack([las.x, las.y, las.z]))
        r, g, b = np.asarray(las.red, dtype=float), np.asarray(las.green, dtype=float), np.asarray(las.blue, dtype=float)
        scale = 65535.0 if max(r.max(), g.max(), b.max()) > 255 else 255.0
        all_rgb.append(np.stack([r, g, b], axis=1) / scale)
        print(f"  loaded {p}: {len(las.points):,} points")
    return np.vstack(all_xyz), np.vstack(all_rgb)


def excess_green_index(rgb01_patch):
    r, g, b = rgb01_patch[..., 0] * 255, rgb01_patch[..., 1] * 255, rgb01_patch[..., 2] * 255
    return 2 * g - r - b


def rasterize_local_patch(xyz, rgb01, cx, cy, half, res=0.015):
    """Builds a small, high-resolution local RGB image (topmost point's
    color per cell, with nearest-neighbor gap-filling) for one tree's
    immediate neighborhood. res=1.5cm matches the real point spacing
    established earlier in this project (median ~1-2cm) -- an earlier
    version of this function used 5mm cells, FINER than the real point
    spacing, which left most cells empty (black) and made the image
    unreadable; confirmed by actually looking at the output, not just
    checking that the script ran without error."""
    dxy_x = np.abs(xyz[:, 0] - cx)
    dxy_y = np.abs(xyz[:, 1] - cy)
    nearby = (dxy_x <= half) & (dxy_y <= half)
    if not nearby.any():
        return None

    local_xyz = xyz[nearby]
    local_rgb = rgb01[nearby]

    n_px = max(2, int(2 * half / res) + 1)
    col = np.clip(((local_xyz[:, 0] - (cx - half)) / res).astype(int), 0, n_px - 1)
    row = np.clip((((cy + half) - local_xyz[:, 1]) / res).astype(int), 0, n_px - 1)
    z = local_xyz[:, 2]

    rgb_img = np.zeros((n_px, n_px, 3))
    has_data = np.zeros((n_px, n_px), dtype=bool)
    order = np.argsort(z)  # topmost point per cell wins (last write in sorted-ascending order)
    flat_idx = row[order] * n_px + col[order]
    flat_rgb = rgb_img.reshape(-1, 3)
    flat_has = has_data.reshape(-1)
    flat_rgb[flat_idx] = local_rgb[order]
    flat_has[flat_idx] = True

    # Gap-fill empty cells from their nearest populated neighbor -- without
    # this, most cells stay black and the image is unreadable, exactly
    # what the first test run showed.
    if (~has_data).any() and has_data.any():
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(~has_data, return_distances=True, return_indices=True)
        rgb_img = rgb_img[indices[0], indices[1]]

    return rgb_img


def load_fallback_trees(csv_path, methods):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["method"] in methods:
                rows.append(r)
    return rows


# ================================================================================
# MAIN
# ================================================================================

def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("[1/3] Reading Stage 1 output, finding fallback trees...")
    fallback_rows = load_fallback_trees(cfg["stage1_csv"], cfg["fallback_methods"])
    print(f"  found {len(fallback_rows)} fallback trees: "
          f"{sum(1 for r in fallback_rows if r['method']=='fallback_zero_score')} zero_score, "
          f"{sum(1 for r in fallback_rows if r['method']=='fallback_no_height_bump')} no_height_bump")

    if not fallback_rows:
        print("  No fallback trees found -- nothing to diagnose. Every tree localized normally.")
        return

    print("[2/3] Loading all LAS points once...")
    xyz, rgb01 = load_all_points(cfg["las_files"])

    print(f"[3/3] Building diagnostic crops for {len(fallback_rows)} trees...")
    half = cfg["view_radius_m"]
    n_per_image = cfg["max_trees_per_image"]
    n_images = int(np.ceil(len(fallback_rows) / n_per_image))

    for img_i in range(n_images):
        batch = fallback_rows[img_i * n_per_image: (img_i + 1) * n_per_image]
        ncols = 4
        nrows = int(np.ceil(len(batch) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)

        for i, r in enumerate(batch):
            tid, tx, ty = r["tree_id"], float(r["orig_x"]), float(r["orig_y"])
            rgb_img = rasterize_local_patch(xyz, rgb01, tx, ty, half)
            ax = axes[i // ncols, i % ncols]
            if rgb_img is not None:
                ax.imshow(np.clip(rgb_img, 0, 1), extent=(-half, half, -half, half), origin="upper")
            else:
                ax.set_facecolor("black")
            ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2)
            circ = plt.Circle((0, 0), cfg["buffer_radius_m"], fill=False, color="yellow", linewidth=0.8)
            ax.add_patch(circ)
            ax.set_title(f"tree {tid}\n{r['method']}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        for j in range(len(batch), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.suptitle("Fallback trees -- red=survey point, yellow=Stage 1 search buffer")
        plt.tight_layout()
        out_path = os.path.join(cfg["output_dir"], f"fallback_diagnostic_{img_i+1}.png")
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"  saved {out_path} ({len(batch)} trees)")

    print(f"\nDone. See {cfg['output_dir']}/ -- look at each panel: is there really "
          f"nothing findable here (genuinely bare/ambiguous ground), or does a plant "
          f"seem visible that the algorithm should have caught?")


if __name__ == "__main__":
    main()
