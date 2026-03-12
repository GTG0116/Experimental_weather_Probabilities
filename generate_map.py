#!/usr/bin/env python3
"""
HRRR Tornado Probability Map Generator
=======================================
Generates 6-hour tornado probability maps for the Northeast US.

Composite index ingredients (all from HRRR SFC files):
  - Surface-Based CAPE              (CAPE:surface)
  - 0-3 km Storm-Relative Helicity  (HLCY:3000-0 m above ground)
  - 0-1 km Storm-Relative Helicity  (HLCY:1000-0 m above ground)  → used in STP
  - 1-hr Max 0-3 km Updraft Helicity(MXUPHL:3000-0 m above ground)
  - 0-6 km Bulk Wind Shear          (VUCSH/VVCSH:0-6000 m above ground)
  - LCL Height                      (HGT:level of adiabatic condensation from sfc)

Significant Tornado Parameter (STP) is computed from the above:
  STP = (SBCAPE/1500) × (SRH₁/150) × (BS₀₋₆/20) × ((2000-LCL)/1000)

Probability represents estimated odds of a tornado within 25 statute miles
of any given point during each forecast hour.
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
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import gaussian_filter
from herbie import Herbie

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NORTHEAST_EXTENT = [-82.5, -64.0, 36.0, 48.5]   # [W, E, S, N]
GRID_SPACING_KM  = 3.0
RADIUS_MILES     = 25.0
RADIUS_KM        = RADIUS_MILES * 1.60934
SIGMA            = RADIUS_KM / GRID_SPACING_KM   # Gaussian sigma in grid cells

# SPC-style probability levels and colors
PROB_LEVELS = [2, 5, 10, 15, 25, 35, 45, 60]
PROB_COLORS = [
    "#01a101",   #  2 %  dark green
    "#79c900",   #  5 %  yellow-green
    "#f5f500",   # 10 %  yellow
    "#e8a400",   # 15 %  orange
    "#fc5d00",   # 25 %  dark orange
    "#e30000",   # 35 %  red
    "#c000c0",   # 45 %  magenta
    "#770077",   # 60 %  dark magenta
]

OUTPUT_DIR = os.path.join("docs", "maps")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.json")


# ---------------------------------------------------------------------------
# HRRR retrieval
# ---------------------------------------------------------------------------
def fetch(H: Herbie, search: str, label: str):
    """Fetch one HRRR field; returns numpy array or None on failure."""
    try:
        ds = H.xarray(search, remove_grib=False)
        varname = list(ds.data_vars)[0]
        return ds[varname]
    except Exception as exc:
        print(f"  [WARN] {label!r} not available ({search}): {exc}")
        return None


def get_fields(run_time: datetime, fxx: int) -> dict:
    """Download all needed HRRR SFC fields for one forecast hour."""
    print(f"  Fetching HRRR {run_time:%Y-%m-%d %H}z F{fxx:02d} …")
    H = Herbie(run_time, model="hrrr", product="sfc", fxx=fxx, verbose=False)

    cape_da  = fetch(H, r"CAPE:surface",                               "SBCAPE")
    srh3_da  = fetch(H, r"HLCY:3000-0 m above ground",                "0-3km SRH")
    srh1_da  = fetch(H, r"HLCY:1000-0 m above ground",                "0-1km SRH")
    uh03_da  = fetch(H, r"MXUPHL:3000-0 m above ground",              "0-3km UH")
    shu_da   = fetch(H, r"VUCSH:0-6000 m above ground",               "BS U")
    shv_da   = fetch(H, r"VVCSH:0-6000 m above ground",               "BS V")
    lcl_da   = fetch(H, r"HGT:level of adiabatic condensation from sfc", "LCL")

    if cape_da is None or uh03_da is None:
        raise RuntimeError(f"Essential fields missing for fxx={fxx}")

    lats = cape_da.latitude.values
    lons = cape_da.longitude.values

    def v(da):
        return da.values.astype(np.float32) if da is not None else None

    cape_np = v(cape_da)
    srh1_np = v(srh1_da)
    shu_np  = v(shu_da)
    shv_np  = v(shv_da)
    lcl_np  = v(lcl_da)

    # Compute Significant Tornado Parameter
    # STP = (SBCAPE/1500) × (SRH₁/150) × (BS₀₋₆/20) × ((2000-LCL)/1000)
    if cape_np is not None and srh1_np is not None and shu_np is not None:
        bs = np.sqrt(shu_np ** 2 + shv_np ** 2)
        stp = (
            np.clip(cape_np, 0, None) / 1500.0 *
            np.clip(srh1_np, 0, None) / 150.0 *
            np.clip(bs,      0, None) / 20.0
        )
        if lcl_np is not None:
            lcl_term = np.clip((2000.0 - lcl_np) / 1000.0, 0.0, 1.5)
            stp = stp * lcl_term
        stp = np.clip(stp, 0, None).astype(np.float32)
    else:
        stp = None

    return {
        "lats":  lats, "lons": lons,
        "cape":  cape_np,
        "srh3":  v(srh3_da),
        "uh03":  np.abs(v(uh03_da)) if uh03_da is not None else None,
        "bs_u":  shu_np, "bs_v": shv_np,
        "stp":   stp,
    }


# ---------------------------------------------------------------------------
# Probability computation
# ---------------------------------------------------------------------------
def compute_prob(fields: dict) -> np.ndarray:
    """
    Weighted composite tornado probability index → 0–95 %.

    Normalization thresholds (where ingredient = 1.0):
      UH 0-3km : 75 m² s⁻²    (cap 4×)
      STP      : 1.0           (cap 4×)
      SRH 0-3km: 250 m² s⁻²   (cap 3×)
      SBCAPE   : 1 500 J kg⁻¹ (cap 3×)
      BS 0-6km : 20 m s⁻¹     (cap 3×)
    """
    shape = fields["cape"].shape
    z     = np.zeros(shape, np.float32)

    uh03 = fields["uh03"] if fields["uh03"] is not None else z
    stp  = fields["stp"]  if fields["stp"]  is not None else z
    srh3 = np.clip(fields["srh3"], 0, None) if fields["srh3"] is not None else z
    cape = np.clip(fields["cape"], 0, None)
    bs   = (np.sqrt(fields["bs_u"]**2 + fields["bs_v"]**2)
            if fields["bs_u"] is not None else z)

    uh_n  = np.clip(uh03 / 75.0,   0, 4.0)
    stp_n = np.clip(stp  / 1.0,    0, 4.0)
    srh_n = np.clip(srh3 / 250.0,  0, 3.0)
    cap_n = np.clip(cape / 1500.0, 0, 3.0)
    bs_n  = np.clip(bs   / 20.0,   0, 3.0)

    composite = (
        0.38 * uh_n  +
        0.28 * stp_n +
        0.18 * srh_n +
        0.10 * cap_n +
        0.06 * bs_n
    )

    # Non-linear UH gate — suppresses false alarms without rotation
    uh_gate  = np.clip(uh03 / 50.0, 0.0, 1.0) ** 0.6
    composite = composite * uh_gate

    prob = 100.0 * (1.0 - np.exp(-0.9 * composite ** 1.6))
    return np.clip(prob, 0, 95.0).astype(np.float32)


def smooth(prob: np.ndarray) -> np.ndarray:
    """Gaussian smoothing representing 25-mile positional uncertainty."""
    return gaussian_filter(prob, sigma=SIGMA)


# ---------------------------------------------------------------------------
# Map rendering  (Pivotal Weather / Tropical Tidbits style)
# ---------------------------------------------------------------------------
def make_map(
    lats, lons, prob,
    run_time: datetime,
    valid_time: datetime,
    fxx: int,
    is_max: bool,
    output_path: str,
):
    W, E, S, N = NORTHEAST_EXTENT
    proj  = ccrs.LambertConformal(central_longitude=-73.5, central_latitude=42.5,
                                   standard_parallels=(33, 45))
    pc    = ccrs.PlateCarree()

    fig   = plt.figure(figsize=(16, 11), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")   # dark navy page background

    # Map axes: leave room at top for header strip, bottom for colorbar + caption
    ax = fig.add_axes([0.01, 0.10, 0.98, 0.84], projection=proj)
    ax.set_extent([W, E, S, N], crs=pc)
    ax.set_facecolor("#1c3557")          # deep blue-gray ocean default

    # ---- Background features (land / water / borders) --------------------
    LAND   = cfeature.NaturalEarthFeature("physical", "land",          "50m")
    OCEAN  = cfeature.NaturalEarthFeature("physical", "ocean",         "50m")
    LAKES  = cfeature.NaturalEarthFeature("physical", "lakes",         "50m")
    RIVERS = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m")
    STATES = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m")
    COUNTIES = cfeature.NaturalEarthFeature("cultural", "admin_2_counties", "10m")
    BORDERS  = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land",  "50m")
    COAST    = cfeature.NaturalEarthFeature("physical", "coastline",   "50m")

    ax.add_feature(OCEAN,   facecolor="#1c3557",   zorder=0)
    ax.add_feature(LAND,    facecolor="#f0ede6",   zorder=1)
    ax.add_feature(LAKES,   facecolor="#1c3557",   zorder=2)
    ax.add_feature(RIVERS,  edgecolor="#7baed4", linewidth=0.3, facecolor="none", zorder=3)
    ax.add_feature(COUNTIES,edgecolor="#bbbbbb", linewidth=0.25,facecolor="none", zorder=4)
    ax.add_feature(STATES,  edgecolor="#555555", linewidth=0.8, facecolor="none", zorder=5)
    ax.add_feature(BORDERS, edgecolor="#222222", linewidth=1.2, facecolor="none", zorder=6)
    ax.add_feature(COAST,   edgecolor="#222222", linewidth=0.7, facecolor="none", zorder=6)

    # ---- Probability fill ------------------------------------------------
    prob_masked = np.ma.masked_less(prob, PROB_LEVELS[0])

    cf = ax.contourf(
        lons, lats, prob_masked,
        levels=PROB_LEVELS,
        colors=PROB_COLORS,
        alpha=0.80,
        transform=pc,
        zorder=7,
        extend="max",
    )
    # Thin black outlines on each contour
    ax.contour(
        lons, lats, prob,
        levels=PROB_LEVELS,
        colors=["#00000055"],
        linewidths=0.5,
        transform=pc,
        zorder=8,
    )

    # ---- Lat/lon gridlines (Pivotal-style) --------------------------------
    gl = ax.gridlines(
        crs=pc, draw_labels=True,
        linewidth=0.5, color="#888888", alpha=0.6, linestyle="--",
        x_inline=False, y_inline=False,
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xlocator      = mticker.FixedLocator(range(-82, -63, 2))
    gl.ylocator      = mticker.FixedLocator(range(36, 50, 2))
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 7.5, "color": "#cccccc"}
    gl.ylabel_style  = {"size": 7.5, "color": "#cccccc"}

    # ---- Colorbar --------------------------------------------------------
    cbar_ax = fig.add_axes([0.12, 0.055, 0.76, 0.020])
    norm    = matplotlib.colors.BoundaryNorm(PROB_LEVELS + [100], len(PROB_COLORS))
    sm      = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.ListedColormap(PROB_COLORS), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                        boundaries=PROB_LEVELS + [100], ticks=PROB_LEVELS)
    cbar.ax.set_xticklabels([f"{v}%" for v in PROB_LEVELS], fontsize=9.5, color="white",
                             fontweight="bold")
    cbar.outline.set_edgecolor("#888888")
    cbar.ax.tick_params(color="#888888", length=4)
    cbar_ax.set_xlabel("Tornado Probability Within 25 Miles", fontsize=9.5,
                        color="white", labelpad=5)

    # ---- Header bar (dark strip) -----------------------------------------
    run_str   = run_time.strftime("%d %b %Y  %H:%M UTC")
    valid_str = valid_time.strftime("%d %b %Y  %H:%M UTC")
    fxx_str   = f"F{fxx:02d}"
    prod_str  = "6-Hour Maximum Envelope" if is_max else f"Hour {fxx_str}"

    title_main = "HRRR  |  Tornado Probability  |  Northeast US"
    title_sub  = f"Run: {run_str}     Valid: {valid_str}     {prod_str}"

    ax.set_title(title_main, loc="left",  fontsize=13, fontweight="bold",
                 color="white", pad=6,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#1a1a2e")])
    ax.set_title(title_sub,  loc="right", fontsize=8.5, color="#dddddd", pad=6,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a2e")])

    # ---- Bottom caption (below colorbar, in figure coords) ---------------
    fig.text(
        0.01, 0.022,
        "Ingredients: SBCAPE · 0–3 km SRH · 1-hr Max 0–3 km UH · "
        "STP (CAPE × SRH₁ × BS₀₋₆ × LCL term) · 0–6 km Bulk Shear",
        fontsize=7.5, color="#aaaaaa", ha="left", va="bottom",
    )
    fig.text(
        0.99, 0.022,
        f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC  |  NOAA HRRR via Herbie",
        fontsize=7.5, color="#aaaaaa", ha="right", va="bottom",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def load_index() -> dict:
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {"runs": []}


def save_index(index: dict):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)


def register_run(index: dict, entry: dict):
    key = entry["run_time"]
    index["runs"] = [r for r in index["runs"] if r["run_time"] != key]
    index["runs"].insert(0, entry)
    index["runs"] = index["runs"][:24]
    index["last_updated"] = datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_run_time(s: str):
    """Parse a run-time string, tolerating empty/None gracefully."""
    if not s or not s.strip():
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    sys.exit(f"[ERROR] Cannot parse --run-time value: {s!r}\n"
             f"  Expected format: 'YYYY-MM-DD HH' or 'YYYY-MM-DD HH:MM'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HRRR-based tornado probability maps for Northeast US")
    parser.add_argument(
        "--run-time", default="",
        help="HRRR model init time (UTC): 'YYYY-MM-DD HH' or 'YYYY-MM-DD HH:MM'. "
             "Leave blank for the most-recent available run.")
    parser.add_argument(
        "--hours", type=int, default=6,
        help="Forecast hours to generate: 1–6 (default 6).")
    args = parser.parse_args()

    run_time = parse_run_time(args.run_time)
    if run_time is None:
        # Default: most-recent HRRR run (~2 h behind now, naive UTC)
        now      = datetime.utcnow()
        run_time = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)

    n_hours  = max(1, min(args.hours, 6))
    run_key  = run_time.strftime("%Y%m%d_%H%M")
    run_dir  = os.path.join(OUTPUT_DIR, run_key)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nHRRR Tornado Probability Map Generator")
    print(f"Run  : {run_time:%Y-%m-%d %H:%M UTC}")
    print(f"Hours: F01 – F{n_hours:02d}\n")

    hourly_probs = []
    hourly_maps  = []

    for fxx in range(1, n_hours + 1):
        valid_time = run_time + timedelta(hours=fxx)
        print(f"\n── F{fxx:02d}  ({valid_time:%H:%M UTC}) ──")
        try:
            fields = get_fields(run_time, fxx)
        except Exception as exc:
            print(f"  [ERROR] Skipping F{fxx:02d}: {exc}")
            continue

        lats, lons = fields["lats"], fields["lons"]
        prob_raw   = compute_prob(fields)
        prob_sm    = smooth(prob_raw)

        hourly_probs.append((valid_time, prob_sm, lats, lons))

        fname = f"tornado_prob_F{fxx:02d}.png"
        fpath = os.path.join(run_dir, fname)
        make_map(lats, lons, prob_sm, run_time, valid_time,
                 fxx=fxx, is_max=False, output_path=fpath)
        hourly_maps.append({
            "fxx": fxx,
            "valid_time": valid_time.isoformat(),
            "file": f"{run_key}/{fname}",
        })

    if not hourly_probs:
        sys.exit("[ERROR] No forecast hours processed — aborting.")

    # 6-hour maximum envelope
    print(f"\n── 6-hr Maximum Envelope ──")
    all_probs = np.stack([p for _, p, _, _ in hourly_probs], axis=0)
    prob_max  = np.max(all_probs, axis=0)
    lats_r, lons_r = hourly_probs[0][2], hourly_probs[0][3]
    valid_end = hourly_probs[-1][0]

    max_fname = "tornado_prob_MAX.png"
    max_fpath = os.path.join(run_dir, max_fname)
    make_map(lats_r, lons_r, prob_max, run_time, valid_end,
             fxx=n_hours, is_max=True, output_path=max_fpath)

    # Update index
    index = load_index()
    register_run(index, {
        "run_time":      run_time.isoformat(),
        "run_key":       run_key,
        "n_hours":       n_hours,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "max_prob_map":  f"{run_key}/{max_fname}",
        "hourly_maps":   hourly_maps,
    })
    save_index(index)
    print(f"\nIndex updated → {INDEX_FILE}")

    # latest.json for web viewer
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump({"run_key": run_key, "run_time": run_time.isoformat()}, f)

    print("Done!\n")


if __name__ == "__main__":
    main()
