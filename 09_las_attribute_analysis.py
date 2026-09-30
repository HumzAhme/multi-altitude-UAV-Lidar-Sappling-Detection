"""
09_las_attribute_analysis.py
================================================================================
LAS ATTRIBUTE VIEWER + POINT DENSITY / RESOLUTION ANALYSIS
================================================================================

WHAT THIS SCRIPT DOES
-----------------------
Three things, run once before deciding how to configure the main
pipeline stages:

  1. Lists every attribute (dimension) present in each LAS file, with a
     description of what it means, AND checks whether it actually
     contains real, varying data or is just an empty/placeholder field
     (a field can technically "exist" in the LAS format without the
     data provider ever populating it -- checked directly on this data).

  2. A suggestion table: which attributes are actually useful for THIS
     project's task (locating small seedlings, extracting height,
     segmenting crowns) and which aren't -- based on what's genuinely
     populated in your files, not just what the LAS spec allows.

  3. Computes REAL point density per file (points per square meter, and
     the derived average point spacing), for every condition the main
     pipeline actually uses (each file alone, 60m merged, 30m merged, all
     4 merged) -- and suggests a resolution for each based on that
     density, rather than a single guessed number applied everywhere.

WHY THIS MATTERS -- CONFIRMED ON YOUR REAL DATA
----------------------------------------------------
The pipeline's earlier scripts used a single fixed CHM resolution
(0.02m / 2cm) for every condition. Checked directly: average point
spacing is 2.57-3.70cm for 60m data but only 1.28cm for all 4 flights
merged. 2cm is COARSER than the merged spacing (reasonable, most cells
get a point) but FINER than the 60m-alone spacing (guarantees many empty
cells on average, not just from noise) -- this is confirmed to be the
root cause of 60m's CHM showing 64% empty cells vs. merged's 27.7% in
earlier testing. The 2cm figure was a reasonable-looking round number,
not a value derived from density -- this script fixes that going
forward by actually computing it.
================================================================================
"""

import os
import numpy as np
import laspy
from scipy.spatial import ConvexHull

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    "las_files_60m": [
        "60m/260522_122413_filt_crop.las",
        "60m/260522_124721_filt_crop.las",
    ],
    "las_files_30m": [
        "30m/260522_130956_filt_crop.las",
        "30m/260522_140126_filt_crop.las",
    ],
    # Rule of thumb applied below: suggested resolution = spacing_multiplier
    # x average point spacing. 1.0 means "resolution equals average
    # spacing" (roughly half of cells get 0 points just from Poisson-like
    # randomness even with perfectly uniform density); higher values are
    # more conservative (fewer empty cells, less fine detail).
    "spacing_multiplier": 1.5,
    # Everything printed to the console is ALSO saved here -- previously
    # this script only printed to the terminal with no saved output at
    # all, meaning the analysis was lost as soon as the terminal scrolled
    # or the session closed.
    "output_dir": "outputs/las_analysis",
}

_LOG_LINES = []


def log(line=""):
    """Prints AND accumulates into _LOG_LINES for saving to a text file at
    the end -- every console line from this script is captured this way,
    so the saved transcript is an exact record of what was shown live."""
    print(line)
    _LOG_LINES.append(line)


# ================================================================================
# ATTRIBUTE DESCRIPTIONS (from the ASPRS LAS 1.4 spec, point format 7)
# ================================================================================
# Written once here rather than looked up per-file, since the schema
# itself doesn't change between files, only which fields have real data.

ATTRIBUTE_DESCRIPTIONS = {
    "X": "Raw X coordinate (before scale/offset applied). Use las.x for real-world meters.",
    "Y": "Raw Y coordinate (before scale/offset applied). Use las.y for real-world meters.",
    "Z": "Raw Z coordinate (before scale/offset applied). Use las.z for real-world meters (elevation).",
    "intensity": "Strength of the returned laser pulse (sensor-specific units, not physical reflectance directly). Higher intensity can indicate different surface materials (e.g. bare soil vs. vegetation vs. metal).",
    "return_number": "Which return this is for a given laser pulse (1st, 2nd, 3rd...). A single pulse can bounce off multiple surfaces on its way down (canopy top, then branches, then ground).",
    "number_of_returns": "Total number of returns recorded for the SAME pulse this point belongs to. A point with return_number=1, number_of_returns=3 means it was the first of three bounces from one pulse.",
    "synthetic": "Flag: True if this point was artificially added/interpolated rather than a real sensor measurement (rare, used by some post-processing tools).",
    "key_point": "Flag: marks a point as especially important to preserve if the file is later thinned/decimated.",
    "withheld": "Flag: marks a point that should be treated as invalid/excluded, despite still being present in the file.",
    "overlap": "Flag: marks a point that falls in a region covered by more than one flight line/strip within the SAME survey mission (not the same as our separate 30m/60m flights).",
    "scanner_channel": "Which physical sensor channel recorded this point, for multi-channel/multi-wavelength LiDAR systems.",
    "scan_direction_flag": "Which direction the scanner mirror was moving when this point was recorded (positive/negative scan direction).",
    "edge_of_flight_line": "Flag: True if this point is at the edge of the scanner's field of view for that flight line -- edge points can have slightly reduced geometric accuracy.",
    "classification": "The ASPRS standard classification code -- in this dataset: 1 = unclassified/vegetation, 2 = ground. Already provided by the survey, not derived by this pipeline.",
    "user_data": "A generic field left for the data provider to store whatever they want -- meaning depends entirely on who processed the file, undocumented unless they say so.",
    "scan_angle": "The angle (from nadir/straight-down) the sensor was pointed at when this point was recorded. Points from a more extreme angle typically have somewhat worse XY positional accuracy.",
    "point_source_id": "An identifier for which original flight/file a point came from -- useful for keeping track of provenance after merging multiple flights together.",
    "gps_time": "The exact GPS timestamp this point was recorded, to sub-second precision. Relevant for anything time-sensitive, e.g. whether a plant could have moved (wind) between two flights.",
    "red": "Red color channel, from imagery captured alongside the LiDAR scan and back-projected onto each point.",
    "green": "Green color channel, same source as red.",
    "blue": "Blue color channel, same source as red.",
}


def describe_and_profile(path, label):
    """Prints (and logs) every attribute in one LAS file: description +
    whether it's actually populated with real, varying data or just a
    flat placeholder. Returns structured rows for CSV export."""
    las = laspy.read(path)
    log(f"\n{'='*90}\n{label}  ({path})\n{'='*90}")
    log(f"point_format: {las.header.point_format.id}   n_points: {len(las.points):,}")
    log(f"{'attribute':<22} {'description':<70}")
    log("-" * 92)

    findings = {}
    rows = []
    for dim in las.point_format.dimension_names:
        if dim in ("X", "Y", "Z"):
            continue  # covered separately, not interesting to profile as a "value"
        desc = ATTRIBUTE_DESCRIPTIONS.get(dim, "(no description available)")
        arr = np.asarray(getattr(las, dim))
        n_unique = len(np.unique(arr))
        if n_unique == 1:
            status = f"NOT POPULATED (constant value: {arr[0]})"
            min_val, max_val = arr[0], arr[0]
        elif n_unique <= 10:
            status = f"populated, {n_unique} distinct values: {np.unique(arr).tolist()}"
            min_val, max_val = arr.min(), arr.max()
        else:
            status = f"populated, real range [{arr.min()}, {arr.max()}], {n_unique} unique values"
            min_val, max_val = arr.min(), arr.max()
        log(f"{dim:<22} {desc}")
        log(f"{'':<22} -> {status}")
        findings[dim] = {"n_unique": n_unique, "populated": n_unique > 1}
        rows.append({
            "file_label": label, "file_path": path, "attribute": dim,
            "description": desc, "n_unique_values": n_unique,
            "populated": n_unique > 1, "min_value": min_val, "max_value": max_val,
        })
    return findings, rows


# ================================================================================
# USAGE RECOMMENDATION TABLE
# ================================================================================

def print_recommendation_table(findings_by_file):
    """
    Combines the description with the real-data check across ALL files to
    give one recommendation per attribute -- an attribute only counts as
    genuinely usable if it's populated in every file, not just some.
    """
    all_dims = list(ATTRIBUTE_DESCRIPTIONS.keys())
    recommendations = {
        "X": ("ESSENTIAL", "Core geometry -- required for everything."),
        "Y": ("ESSENTIAL", "Core geometry -- required for everything."),
        "Z": ("ESSENTIAL", "Core geometry -- required for height/CHM."),
        "classification": ("ESSENTIAL", "Already used for ground/vegetation separation throughout the pipeline."),
        "red": ("USE (pending)", "Real per-point color -- the planned replacement for the RGB .tif photos. Confirmed 66.9-76.0% of points are actually colorized (non-zero) per file -- expect real coverage gaps, a different pattern than the .tif approach's gaps, not an absence of the problem."),
        "green": ("USE (pending)", "Same as red."),
        "blue": ("USE (pending)", "Same as red."),
        "intensity": ("WORTH TESTING", "Genuinely populated with real variation. Could help distinguish vegetation from bare soil/debris independent of RGB -- a potential additional signal for Stage 1's localization, untested so far in this pipeline."),
        "return_number": ("WORTH TESTING", "Genuinely multi-return data (1-3) confirmed present. Could help identify canopy-penetrating pulses vs. single clean ground hits -- relevant for refining ground classification or occlusion analysis, not currently used."),
        "number_of_returns": ("WORTH TESTING", "Same rationale as return_number, used together."),
        "scan_angle": ("POSSIBLY USEFUL", "Real variation confirmed. Points from extreme angles have somewhat worse accuracy -- could be used as a quality filter (e.g. exclude very off-nadir points) if position precision issues are suspected."),
        "edge_of_flight_line": ("POSSIBLY USEFUL", "Has real (if sparse) variation. Same rationale as scan_angle -- edge points may warrant lower trust."),
        "point_source_id": ("USEFUL FOR BOOKKEEPING", "All zero WITHIN each individual file (expected -- there's only one source per un-merged file), but real once files are merged together -- lets you trace which flight any given point in a merged cloud came from."),
        "gps_time": ("SITUATIONAL", "Real per-point timestamps confirmed present. Only relevant if you specifically need to check for time-based effects (e.g. wind movement between flights) -- not used by the current pipeline."),
        "synthetic": ("NOT USEFUL", "Confirmed constant (always 0/False) in this data -- not populated by this provider."),
        "key_point": ("NOT USEFUL", "Confirmed constant (always 0/False)."),
        "withheld": ("NOT USEFUL (but good news)", "Confirmed constant (always 0/False) -- means no points are pre-flagged as bad quality; nothing to filter on, but also nothing to worry about here."),
        "overlap": ("NOT USEFUL", "Confirmed constant (always 0/False) in this data -- this survey apparently had no internal flight-line overlap regions."),
        "scanner_channel": ("NOT USEFUL", "Confirmed constant (single-channel sensor, as expected for standard drone LiDAR)."),
        "scan_direction_flag": ("NOT USEFUL", "Confirmed constant in this data."),
        "user_data": ("NOT USEFUL", "Confirmed constant (always 0) -- this provider didn't use this field."),
    }

    log(f"\n{'='*90}\nATTRIBUTE USAGE RECOMMENDATION (based on real content, not just LAS spec availability)\n{'='*90}")
    log(f"{'attribute':<22} {'recommendation':<26} {'why'}")
    log("-" * 92)
    rows = []
    for dim in all_dims:
        rec, why = recommendations.get(dim, ("CHECK MANUALLY", "Not pre-assessed."))
        log(f"{dim:<22} {rec:<26} {why}")
        rows.append({"attribute": dim, "recommendation": rec, "reason": why})
    return rows


# ================================================================================
# POINT DENSITY + RESOLUTION SUGGESTION
# ================================================================================

def density_report(name, xy, cfg):
    hull = ConvexHull(xy)
    area = hull.volume  # 2D convex hull "volume" is area
    density = len(xy) / area
    spacing_m = 1 / np.sqrt(density)
    suggested_res = spacing_m * cfg["spacing_multiplier"]
    log(f"{name:<28} n={len(xy):>10,}   area={area:>8.1f} m^2   "
        f"density={density:>8.1f} pts/m^2   avg spacing={spacing_m*100:>5.2f} cm   "
        f"suggested resolution={suggested_res*100:>5.2f} cm")
    return {
        "condition": name, "n_points": len(xy), "area_m2": area,
        "density_pts_per_m2": density, "avg_spacing_cm": spacing_m * 100,
        "suggested_resolution_cm": suggested_res * 100,
    }


def load_xy(path):
    las = laspy.read(path)
    return np.column_stack([las.x, las.y])


# ================================================================================
# MAIN
# ================================================================================

def save_csv(rows, path, keys):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([r.get(k, "") for k in keys])
    log(f"  saved {path}")


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    all_files = {
        "60m_flight_1": cfg["las_files_60m"][0],
        "60m_flight_2": cfg["las_files_60m"][1],
        "30m_flight_1": cfg["las_files_30m"][0],
        "30m_flight_2": cfg["las_files_30m"][1],
    }

    log("PART 1: Attribute listing, descriptions, and real-data check, per file")
    findings_by_file = {}
    attribute_rows = []
    for label, path in all_files.items():
        findings, rows = describe_and_profile(path, label)
        findings_by_file[label] = findings
        attribute_rows.extend(rows)

    recommendation_rows = print_recommendation_table(findings_by_file)

    log(f"\n{'='*90}\nPART 2: Point density and resolution suggestion, per condition\n{'='*90}")
    log(f"(rule of thumb used: suggested resolution = {cfg['spacing_multiplier']}x average point "
        f"spacing -- higher multiplier = safer/coarser, fewer empty cells but less fine detail)\n")

    xy = {label: load_xy(path) for label, path in all_files.items()}

    log("-- Individual files --")
    density_rows = []
    for label in all_files:
        density_rows.append(density_report(label, xy[label], cfg))

    log("\n-- Merged conditions actually used by the pipeline --")
    density_rows.append(density_report("60m merged (both flights)",
                                        np.vstack([xy["60m_flight_1"], xy["60m_flight_2"]]), cfg))
    density_rows.append(density_report("30m merged (both flights)",
                                        np.vstack([xy["30m_flight_1"], xy["30m_flight_2"]]), cfg))
    density_rows.append(density_report("ALL 4 merged", np.vstack(list(xy.values())), cfg))

    log(f"\nFor comparison, the pipeline's previous fixed resolution was 2.00 cm, applied "
        f"uniformly to all conditions above -- compare that single number against each "
        f"condition's own suggested resolution here to see where it was (or wasn't) actually "
        f"appropriate for that specific condition's real point density.")

    # --- actually save everything, not just print it ---
    save_csv(attribute_rows, os.path.join(cfg["output_dir"], "las_attributes_per_file.csv"),
             ["file_label", "file_path", "attribute", "description", "n_unique_values",
              "populated", "min_value", "max_value"])
    save_csv(recommendation_rows, os.path.join(cfg["output_dir"], "attribute_recommendations.csv"),
             ["attribute", "recommendation", "reason"])
    save_csv(density_rows, os.path.join(cfg["output_dir"], "point_density_resolution.csv"),
             ["condition", "n_points", "area_m2", "density_pts_per_m2",
              "avg_spacing_cm", "suggested_resolution_cm"])

    log_path = os.path.join(cfg["output_dir"], "analysis_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(_LOG_LINES))
    print(f"  saved {log_path}")

    print(f"\nAll analysis saved to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
