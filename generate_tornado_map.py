#!/usr/bin/env python3
"""
HRRR-Based Tornado Probability Map Generator
=============================================
Generates 6-hour tornado probability maps for the Northeast US.

Composite index uses:
  - Surface-based CAPE
  - 0-6km bulk wind shear magnitude
  - 0-3km Storm-Relative Helicity (SRH)
  - Significant Tornado Parameter (STP)
  - 1-hour max 0-3km AGL Updraft Helicity (UH)

Probability represents odds of a tornado within a 25-mile radius
of any given point over the forecast period.
"""

import os
import sys
import json
import warnings
import argparse
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from herbie import Herbie

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants / Configuration
# ---------------------------------------------------------------------------
NORTHEAST_EXTENT = [-82.5, -64.5, 36.0, 48.5]   # [W, E, S, N]
GRID_SPACING_KM  = 3.0                             # HRRR native ~3 km
RADIUS_KM        = 40.2                            # 25 statute miles ≈ 40.2 km
SIGMA            = RADIUS_KM / GRID_SPACING_KM     # Gaussian sigma in grid cells

# Probability contour levels and matching fill colors
PROB_LEVELS = [2, 5, 10, 15, 25, 35, 45, 60]
PROB_COLORS = [
    "#00bb00",   #  2%  green
    "#77cc00",   #  5%  yellow-green
    "#e6e600",   # 10%  yellow
    "#ffaa00",   # 15%  orange
    "#ff5500",   # 25%  red-orange
    "#cc0000",   # 35%  red
    "#880000",   # 45%  dark red
    "#4a0080",   # 60%  purple
]

OUTPUT_DIR = os.path.join("docs", "maps")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.json")


# ---------------------------------------------------------------------------
# HRRR Data Retrieval
# ---------------------------------------------------------------------------
def fetch_hrrr_field(H: Herbie, search_str: str, field_name: str):
    """Download a single HRRR field; return DataArray or None on failure."""
    try:
        ds = H.xarray(search_str, remove_grib=False)
        # herbie returns a Dataset – grab the first data variable
        varname = list(ds.data_vars)[0]
        return ds[varname]
    except Exception as exc:
        print(f"  [WARN] Could not fetch {field_name!r} ({search_str}): {exc}")
        return None


def get_hrrr_fields(run_time: datetime, fxx: int):
    """
    Return a dict of 2-D numpy arrays for one HRRR run / forecast hour.

    Keys: cape, srh3, uh03, shear_u, shear_v, stp
    Values: masked arrays on HRRR native grid; lat/lon also returned.
    """
    print(f"  Fetching HRRR {run_time:%Y-%m-%d %H}z F{fxx:02d} …")
    H = Herbie(run_time, model="hrrr", product="sfc", fxx=fxx, verbose=False)

    # Fetch fields ---------------------------------------------------------
    cape_da   = fetch_hrrr_field(H, r"CAPE:surface",                   "SBCAPE")
    srh3_da   = fetch_hrrr_field(H, r"HLCY:3000-0 m above ground",     "0-3km SRH")
    srh1_da   = fetch_hrrr_field(H, r"HLCY:1000-0 m above ground",     "0-1km SRH")
    uh03_da   = fetch_hrrr_field(H, r"MXUPHL:3000-0 m above ground",   "0-3km UH")
    shear_u   = fetch_hrrr_field(H, r"VUCSH:0-6000 m above ground",    "0-6km Shear U")
    shear_v   = fetch_hrrr_field(H, r"VVCSH:0-6000 m above ground",    "0-6km Shear V")
    # SIGTOR is not in HRRR SFC files — compute simplified STP below

    # We need at minimum CAPE and UH for a meaningful product
    if cape_da is None or uh03_da is None:
        raise RuntimeError(f"Essential fields missing for fxx={fxx}")

    # Extract lats/lons from one DataArray
    ref_da = cape_da
    lats = ref_da.latitude.values
    lons = ref_da.longitude.values

    def to_np(da):
        return da.values if da is not None else None

    cape_np    = to_np(cape_da)
    srh1_np    = to_np(srh1_da)
    shear_u_np = to_np(shear_u)
    shear_v_np = to_np(shear_v)

    # Simplified Significant Tornado Parameter
    # STP = (SBCAPE/1500) × (0-1km SRH/150) × (0-6km BS / 12 m s⁻¹)
    # Clipped to ≥ 0 (negative SRH/CAPE produce no tornado threat)
    if cape_np is not None and srh1_np is not None and shear_u_np is not None:
        bs_mag = np.sqrt(shear_u_np ** 2 + shear_v_np ** 2)
        stp_np = (
            np.clip(cape_np, 0, None) / 1500.0 *
            np.clip(srh1_np, 0, None) / 150.0 *
            np.clip(bs_mag,  0, None) / 12.0
        )
        stp_np = np.clip(stp_np, 0, None).astype(np.float32)
    else:
        stp_np = None

    return {
        "lats":    lats,
        "lons":    lons,
        "cape":    cape_np,
        "srh3":    to_np(srh3_da),
        "uh03":    to_np(uh03_da),
        "shear_u": shear_u_np,
        "shear_v": shear_v_np,
        "stp":     stp_np,
    }


# ---------------------------------------------------------------------------
# Probability Computation
# ---------------------------------------------------------------------------
def compute_probability(fields: dict) -> np.ndarray:
    """
    Derive a tornado probability index (0–100 %) from HRRR fields.

    Parameterisation
    ----------------
    Each ingredient is normalised to a 0-to-1 (or higher) scale
    where 1 ≈ "very favorable" threshold:

      CAPE      :  normalised by 1 500 J kg⁻¹   (cap 3×)
      SRH 0-3km:  normalised by 250 m² s⁻²       (cap 3×)
      0-3km UH  :  normalised by 75  m² s⁻²       (cap 4×)
      0-6km BS  :  normalised by 20  m s⁻¹ ≈ 39 kt (cap 3×)
      STP       :  normalised by 1.0              (cap 4×)

    Weighted composite → logistic-like probability.
    """
    shape  = fields["cape"].shape
    zeros  = np.zeros(shape, dtype=np.float32)

    cape   = np.where(fields["cape"]    is not None, fields["cape"],    zeros).astype(np.float32)
    srh3   = np.where(fields["srh3"]    is not None, fields["srh3"],    zeros).astype(np.float32)
    uh03   = np.where(fields["uh03"]    is not None, np.abs(fields["uh03"]), zeros).astype(np.float32)
    stp    = np.zeros(shape, dtype=np.float32) if fields["stp"] is None else np.clip(fields["stp"], 0, None).astype(np.float32)

    if fields["shear_u"] is not None and fields["shear_v"] is not None:
        bs = np.sqrt(fields["shear_u"] ** 2 + fields["shear_v"] ** 2).astype(np.float32)
    else:
        bs = zeros

    # Normalise
    cape_n  = np.clip(cape  / 1500.0, 0, 3.0)
    srh3_n  = np.clip(srh3  / 250.0,  0, 3.0)
    uh03_n  = np.clip(uh03  / 75.0,   0, 4.0)
    bs_n    = np.clip(bs    / 20.0,   0, 3.0)
    stp_n   = np.clip(stp   / 1.0,    0, 4.0)

    # Weighted sum (weights sum to 1.0)
    #   UH    is the most direct indicator of mesocyclone rotation
    #   STP   is a proven composite parameter
    #   SRH   contributes to storm-relative inflow
    #   CAPE  provides instability
    #   Shear provides organisational potential
    composite = (
        0.38 * uh03_n  +
        0.28 * stp_n   +
        0.18 * srh3_n  +
        0.10 * cape_n  +
        0.06 * bs_n
    )

    # UH gate: very little probability without meaningful rotation
    uh_gate = np.clip(uh03 / 50.0, 0, 1.0)
    composite = composite * uh_gate

    # Logistic-style mapping  P = 100 × (1 - exp(-k × C^α))
    prob = 100.0 * (1.0 - np.exp(-0.9 * composite ** 1.6))
    prob = np.clip(prob, 0, 95.0)

    return prob.astype(np.float32)


def smooth_prob(prob: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """Apply Gaussian smoothing to simulate 25-mile radius uncertainty."""
    return gaussian_filter(prob, sigma=sigma)


# ---------------------------------------------------------------------------
# Subsetting to Northeast
# ---------------------------------------------------------------------------
def subset_to_northeast(lats, lons, data):
    """Return boolean 2-D mask that selects the Northeast extent."""
    w, e, s, n = NORTHEAST_EXTENT
    mask = (lats >= s) & (lats <= n) & (lons >= w) & (lons <= e)
    return mask


# ---------------------------------------------------------------------------
# Map Plotting
# ---------------------------------------------------------------------------
def make_map(
    lats, lons, prob,
    run_time: datetime,
    valid_time: datetime,
    fxx: int,
    output_path: str,
):
    """Render and save a single tornado-probability PNG."""
    w, e, s, n = NORTHEAST_EXTENT

    proj = ccrs.LambertConformal(central_longitude=-73.5, central_latitude=42.5)
    fig = plt.figure(figsize=(14, 9), dpi=130)
    ax  = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([w, e, s, n], crs=ccrs.PlateCarree())

    # --- Background features
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),    facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),     facecolor="#f5f0e8", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),    facecolor="#d0e8f5", zorder=0)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"),   edgecolor="#93c4d4", linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.STATES.with_scale("50m"),   edgecolor="#777777", linewidth=0.8,  zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),  edgecolor="#444444", linewidth=1.2,  zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),edgecolor="#444444", linewidth=0.8,  zorder=2)

    # --- Probability fill
    prob_masked = np.ma.masked_less(prob, PROB_LEVELS[0])
    cf = ax.contourf(
        lons, lats, prob_masked,
        levels=PROB_LEVELS,
        colors=PROB_COLORS,
        alpha=0.72,
        transform=ccrs.PlateCarree(),
        zorder=3,
        extend="max",
    )

    # --- Probability contour outlines
    ax.contour(
        lons, lats, prob,
        levels=PROB_LEVELS,
        colors="black",
        linewidths=0.4,
        alpha=0.45,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # --- Colourbar
    cbar = plt.colorbar(
        cf, ax=ax,
        orientation="horizontal",
        fraction=0.038, pad=0.03, aspect=45,
        ticks=PROB_LEVELS,
    )
    cbar.set_label("Tornado Probability Within 25 Miles (%)", fontsize=11, labelpad=6)
    cbar.ax.set_xticklabels([f"{v}%" for v in PROB_LEVELS], fontsize=9)

    # --- Title
    run_str   = run_time.strftime("%d %b %Y %H:%M UTC")
    valid_str = valid_time.strftime("%d %b %Y %H:%M UTC")
    ax.set_title(
        f"HRRR Tornado Probability  •  Northeast US\n"
        f"Run: {run_str}   |   Valid: {valid_str}   |   F{fxx:02d}",
        fontsize=13, fontweight="bold", pad=10,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    # --- Info annotation
    ax.text(
        0.01, 0.01,
        "Index: SBCAPE + 0–6 km Shear + 0–3 km SRH + STP + 0–3 km UH  |  25-mi Gaussian radius",
        transform=ax.transAxes,
        fontsize=7.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="#aaaaaa"),
        zorder=10,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Index JSON helpers
# ---------------------------------------------------------------------------
def load_index() -> dict:
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {"runs": []}


def save_index(index: dict):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)


def register_run(index: dict, run_entry: dict):
    """Add or replace a run entry, keeping at most 24 runs."""
    run_key = run_entry["run_time"]
    index["runs"] = [r for r in index["runs"] if r["run_time"] != run_key]
    index["runs"].insert(0, run_entry)
    index["runs"] = index["runs"][:24]
    index["last_updated"] = datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate HRRR tornado probability maps")
    parser.add_argument(
        "--run-time", default=None,
        help="HRRR model run time (UTC) as 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH'. "
             "Defaults to the most-recent available run (~2 h ago).",
    )
    parser.add_argument(
        "--hours", type=int, default=6,
        help="Number of forecast hours (1–6, default 6).",
    )
    args = parser.parse_args()

    # Resolve run time
    if args.run_time:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%dT%H:%M"):
            try:
                # Keep naive UTC — Herbie doesn't handle tz-aware datetimes
                run_time = datetime.strptime(args.run_time, fmt)
                break
            except ValueError:
                pass
        else:
            sys.exit(f"Could not parse --run-time: {args.run_time!r}")
    else:
        # Default: last 00-min mark at least 2 h behind now (naive UTC for Herbie)
        now = datetime.utcnow()
        run_time = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)

    n_hours = max(1, min(args.hours, 6))
    print(f"\nHRRR Tornado Probability Map Generator")
    print(f"Run time : {run_time:%Y-%m-%d %H:%M UTC}")
    print(f"Forecast hours: F01 – F{n_hours:02d}\n")

    run_key = run_time.strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(OUTPUT_DIR, run_key)
    os.makedirs(run_dir, exist_ok=True)

    hourly_probs  = []   # list of (valid_time, prob_array, lats, lons)
    hourly_maps   = []   # list of relative-path strings for index.json

    # -----------------------------------------------------------------------
    # Loop over forecast hours
    # -----------------------------------------------------------------------
    for fxx in range(1, n_hours + 1):
        valid_time = run_time + timedelta(hours=fxx)  # naive UTC
        print(f"\n── F{fxx:02d} ({valid_time:%H:%M UTC}) ──")

        try:
            fields = get_hrrr_fields(run_time, fxx)
        except Exception as exc:
            print(f"  [ERROR] Skipping F{fxx:02d}: {exc}")
            continue

        lats = fields["lats"]
        lons = fields["lons"]

        # Compute probability
        prob_raw  = compute_probability(fields)
        prob_smth = smooth_prob(prob_raw)

        hourly_probs.append((valid_time, prob_smth, lats, lons))

        # Subset display to Northeast
        # (contourf handles out-of-extent automatically via map projection)

        fname = f"tornado_prob_F{fxx:02d}.png"
        fpath = os.path.join(run_dir, fname)
        make_map(lats, lons, prob_smth, run_time, valid_time, fxx, fpath)
        hourly_maps.append({
            "fxx": fxx,
            "valid_time": valid_time.isoformat(),
            "file": f"{run_key}/{fname}",
        })

    if not hourly_probs:
        sys.exit("No forecast hours successfully processed – aborting.")

    # -----------------------------------------------------------------------
    # 6-hour maximum probability (envelope)
    # -----------------------------------------------------------------------
    print("\n── Generating 6-hour maximum probability map ──")
    all_probs = np.stack([p for _, p, _, _ in hourly_probs], axis=0)
    prob_max  = np.max(all_probs, axis=0)
    lats_ref, lons_ref = hourly_probs[0][2], hourly_probs[0][3]

    valid_end = hourly_probs[-1][0]
    max_fname = "tornado_prob_MAX.png"
    max_fpath = os.path.join(run_dir, max_fname)
    make_map(
        lats_ref, lons_ref, prob_max,
        run_time, valid_end,
        fxx=n_hours,
        output_path=max_fpath,
    )
    # Patch max-map title label
    # (title is baked into the PNG; update index entry description instead)

    # -----------------------------------------------------------------------
    # Update index.json
    # -----------------------------------------------------------------------
    index = load_index()
    run_entry = {
        "run_time":       run_time.isoformat(),
        "run_key":        run_key,
        "n_hours":        n_hours,
        "generated_utc":  datetime.utcnow().isoformat() + "Z",
        "max_prob_map":   f"{run_key}/{max_fname}",
        "hourly_maps":    hourly_maps,
    }
    register_run(index, run_entry)
    save_index(index)
    print(f"\nIndex updated → {INDEX_FILE}")

    # -----------------------------------------------------------------------
    # Update docs/index.html pointer (writes latest run key for the webpage)
    # -----------------------------------------------------------------------
    latest_path = os.path.join("docs", "latest.json")
    with open(latest_path, "w") as f:
        json.dump({"run_key": run_key, "run_time": run_time.isoformat()}, f)
    print(f"Latest pointer → {latest_path}")

    print("\nDone!\n")


if __name__ == "__main__":
    main()
