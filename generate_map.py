#!/usr/bin/env python3
"""
Multi-Model Tornado Probability Map Generator
=============================================
Generates 6-hour tornado probability maps for the Northeast US or CONUS
using HRRR, RAP, NAM, or an ensemble combination of all three.

Supported models (--model flag):
  hrrr     : High-Resolution Rapid Refresh (3-km)   – original behaviour
  rap      : Rapid Refresh (13-km)
  nam      : NAM CONUS Nest (3-km convection-allowing)
  combined : Ensemble average of HRRR + RAP + NAM, plus individual maps

Composite index ingredients (GRIB2 SFC fields):
  - Surface-Based CAPE              (CAPE:surface)
  - 0-3 km Storm-Relative Helicity  (HLCY:3000-0 m above ground)
  - 0-1 km Storm-Relative Helicity  (HLCY:1000-0 m above ground) → used in STP
  - 1-hr Max 0-3 km Updraft Helicity(MXUPHL:3000-0 m above ground)
  - 0-6 km Bulk Wind Shear          (VUCSH/VVCSH:0-6000 m above ground)
  - LCL Height                      (HGT:level of adiabatic condensation from sfc)

Significant Tornado Parameter (STP):
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
from scipy.interpolate import griddata
from herbie import Herbie

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NORTHEAST_EXTENT = [-82.5, -64.0, 36.0, 48.5]   # [W, E, S, N]
CONUS_EXTENT     = [-125.0, -66.0, 24.0, 50.0]  # [W, E, S, N]
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
# Model configuration
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "hrrr": {
        "herbie_model": "hrrr",
        "product":      "sfc",
        "label":        "HRRR",
        "label_long":   "HRRR 3-km",
        "grid_km":      3.0,
        "color":        "#4e9af1",   # blue accent for combined map annotation
    },
    "rap": {
        "herbie_model": "rap",
        "product":      None,        # Herbie selects the default RAP product
        "label":        "RAP",
        "label_long":   "RAP 13-km",
        "grid_km":      13.0,
        "color":        "#f1a44e",   # orange
    },
    "nam": {
        "herbie_model": "nam",
        "product":      "conusnest", # 3-km CONUS convection-allowing nest
        "label":        "NAM",
        "label_long":   "NAM-CONUS 3-km",
        "grid_km":      3.0,
        "color":        "#4ef1a4",   # green
    },
}

# Models that form the combined ensemble
ENSEMBLE_MEMBERS = ["hrrr", "rap", "nam"]

# Common-grid resolution for the combined ensemble interpolation
COMMON_GRID_DEG_CONUS = 0.10   # ° (~11 km at mid-latitudes)
COMMON_GRID_DEG_NE    = 0.05   # ° (~5.5 km)
COMMON_GRID_KM        = 11.0   # approximate — used for sigma on the combined grid


# ---------------------------------------------------------------------------
# Herbie retrieval helpers
# ---------------------------------------------------------------------------
def _herbie_instance(run_time: datetime, model: str, fxx: int) -> Herbie:
    cfg = MODEL_CONFIGS[model]
    kwargs = dict(
        date=run_time,
        model=cfg["herbie_model"],
        fxx=fxx,
        verbose=False,
    )
    if cfg["product"] is not None:
        kwargs["product"] = cfg["product"]
    return Herbie(**kwargs)


def fetch(H: Herbie, search: str, label: str):
    """Fetch one field from a Herbie object; returns DataArray or None."""
    try:
        ds = H.xarray(search, remove_grib=False)
        varname = list(ds.data_vars)[0]
        return ds[varname]
    except Exception as exc:
        print(f"  [WARN] {label!r} not available ({search}): {exc}")
        return None


def get_fields(run_time: datetime, fxx: int, model: str = "hrrr") -> dict:
    """Download all needed SFC fields for one model / forecast hour."""
    cfg = MODEL_CONFIGS[model]
    print(f"  Fetching {cfg['label']} {run_time:%Y-%m-%d %H}z F{fxx:02d} …")
    H = _herbie_instance(run_time, model, fxx)

    cape_da = fetch(H, r"CAPE:surface",                                "SBCAPE")
    srh3_da = fetch(H, r"HLCY:3000-0 m above ground",                 "0-3km SRH")
    srh1_da = fetch(H, r"HLCY:1000-0 m above ground",                 "0-1km SRH")
    uh03_da = fetch(H, r"MXUPHL:3000-0 m above ground",               "0-3km UH")
    shu_da  = fetch(H, r"VUCSH:0-6000 m above ground",                "BS U")
    shv_da  = fetch(H, r"VVCSH:0-6000 m above ground",                "BS V")
    lcl_da  = fetch(H, r"HGT:level of adiabatic condensation from sfc", "LCL")

    if cape_da is None:
        raise RuntimeError(f"Essential field CAPE missing for {model} fxx={fxx}")

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
    if cape_np is not None and srh1_np is not None and shu_np is not None and shv_np is not None:
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

    uh03_np = v(uh03_da)
    return {
        "lats": lats, "lons": lons,
        "cape": cape_np,
        "srh3": v(srh3_da),
        "uh03": np.abs(uh03_np) if uh03_np is not None else None,
        "bs_u": shu_np, "bs_v": shv_np,
        "stp":  stp,
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
    # If UH is completely unavailable, release the gate
    if fields["uh03"] is None:
        uh_gate = np.ones(shape, np.float32)
    composite = composite * uh_gate

    prob = 100.0 * (1.0 - np.exp(-0.9 * composite ** 1.6))
    return np.clip(prob, 0, 95.0).astype(np.float32)


def smooth(prob: np.ndarray, model: str = "hrrr") -> np.ndarray:
    """Gaussian smoothing representing 25-mile positional uncertainty."""
    grid_km = MODEL_CONFIGS[model]["grid_km"]
    sigma   = RADIUS_KM / grid_km
    return gaussian_filter(prob, sigma=sigma)


# ---------------------------------------------------------------------------
# Combined ensemble helpers
# ---------------------------------------------------------------------------
def make_common_grid(region: str):
    """Return (lats2d, lons2d) on a regular lat/lon grid for the region."""
    if region == "conus":
        W, E, S, N = CONUS_EXTENT
        step = COMMON_GRID_DEG_CONUS
    else:
        W, E, S, N = NORTHEAST_EXTENT
        step = COMMON_GRID_DEG_NE
    lon1d = np.arange(W, E + step * 0.5, step, dtype=np.float32)
    lat1d = np.arange(S, N + step * 0.5, step, dtype=np.float32)
    lons2d, lats2d = np.meshgrid(lon1d, lat1d)
    return lats2d, lons2d


def interp_to_grid(src_lats, src_lons, src_data, tgt_lats, tgt_lons):
    """
    Interpolate src_data (on a possibly irregular grid) onto the regular
    target lat/lon grid using linear griddata.  Fills outside convex hull
    with 0.0.
    """
    pts  = np.column_stack([src_lats.ravel(), src_lons.ravel()])
    vals = src_data.ravel().astype(np.float64)
    valid = np.isfinite(vals)
    if valid.sum() < 4:
        return np.zeros_like(tgt_lats, dtype=np.float32)
    tgt_pts = np.column_stack([tgt_lats.ravel(), tgt_lons.ravel()])
    result = griddata(pts[valid], vals[valid], tgt_pts,
                      method="linear", fill_value=0.0)
    return result.reshape(tgt_lats.shape).astype(np.float32)


def get_model_prob(run_time: datetime, fxx: int, model: str) -> tuple:
    """
    Fetch fields, compute probability, and smooth for one model/hour.
    Returns (lats, lons, prob_smoothed) or raises on failure.
    """
    fields   = get_fields(run_time, fxx, model)
    prob_raw = compute_prob(fields)
    prob_sm  = smooth(prob_raw, model)
    return fields["lats"], fields["lons"], prob_sm


def combined_prob(run_time: datetime, fxx: int, region: str):
    """
    Fetch HRRR, RAP, and NAM for one forecast hour, interpolate each to a
    common regular grid, and return the (equally-weighted) ensemble mean.

    Returns:
        common_lats, common_lons, prob_combined, per_model dict
        per_model: { model_name: (lats, lons, prob_on_native_grid) }
    """
    common_lats, common_lons = make_common_grid(region)
    stack      = []
    per_model  = {}

    for mdl in ENSEMBLE_MEMBERS:
        try:
            lats_m, lons_m, prob_m = get_model_prob(run_time, fxx, mdl)
            # Interpolate to common grid
            prob_common = interp_to_grid(lats_m, lons_m, prob_m,
                                         common_lats, common_lons)
            stack.append(prob_common)
            per_model[mdl] = (lats_m, lons_m, prob_m)
            print(f"    {MODEL_CONFIGS[mdl]['label']:6s}  max prob = "
                  f"{prob_m.max():.1f}%  (on native grid)")
        except Exception as exc:
            print(f"  [WARN] {mdl.upper()} failed for F{fxx:02d}: {exc}")

    if not stack:
        raise RuntimeError(f"All ensemble members failed for fxx={fxx}")

    prob_ensemble = np.mean(stack, axis=0).astype(np.float32)

    # Light additional smoothing on the common grid (half a grid cell sigma)
    sigma_common = RADIUS_KM / COMMON_GRID_KM
    prob_ensemble = gaussian_filter(prob_ensemble, sigma=sigma_common)
    prob_ensemble = np.clip(prob_ensemble, 0, 95.0).astype(np.float32)

    return common_lats, common_lons, prob_ensemble, per_model


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
    region: str = "conus",
    model: str = "hrrr",
    ensemble_members: list = None,   # list of model names used in combined
):
    if region == "conus":
        W, E, S, N = CONUS_EXTENT
        proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5,
                                     standard_parallels=(33, 45))
        figsize       = (22, 13)
        x_ticks       = list(range(-125, -65, 5))
        y_ticks       = list(range(24, 51, 4))
        region_label  = "CONUS"
    else:
        W, E, S, N = NORTHEAST_EXTENT
        proj = ccrs.LambertConformal(central_longitude=-73.5, central_latitude=42.5,
                                     standard_parallels=(33, 45))
        figsize       = (16, 11)
        x_ticks       = list(range(-82, -63, 2))
        y_ticks       = list(range(36, 50, 2))
        region_label  = "Northeast US"

    pc = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("#1a1a2e")

    ax = fig.add_axes([0.01, 0.10, 0.98, 0.84], projection=proj)
    ax.set_extent([W, E, S, N], crs=pc)
    ax.set_facecolor("#1c3557")

    # Background features
    LAND   = cfeature.NaturalEarthFeature("physical", "land",          "50m")
    OCEAN  = cfeature.NaturalEarthFeature("physical", "ocean",         "50m")
    LAKES  = cfeature.NaturalEarthFeature("physical", "lakes",         "50m")
    RIVERS = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m")
    STATES = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m")
    COUNTIES = cfeature.NaturalEarthFeature("cultural", "admin_2_counties", "10m")
    BORDERS  = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land",  "50m")
    COAST    = cfeature.NaturalEarthFeature("physical", "coastline",   "50m")

    ax.add_feature(OCEAN,    facecolor="#1c3557",   zorder=0)
    ax.add_feature(LAND,     facecolor="#f0ede6",   zorder=1)
    ax.add_feature(LAKES,    facecolor="#1c3557",   zorder=2)
    ax.add_feature(RIVERS,   edgecolor="#7baed4", linewidth=0.3, facecolor="none", zorder=3)
    ax.add_feature(COUNTIES, edgecolor="#bbbbbb", linewidth=0.25,facecolor="none", zorder=4)
    ax.add_feature(STATES,   edgecolor="#555555", linewidth=0.8, facecolor="none", zorder=5)
    ax.add_feature(BORDERS,  edgecolor="#222222", linewidth=1.2, facecolor="none", zorder=6)
    ax.add_feature(COAST,    edgecolor="#222222", linewidth=0.7, facecolor="none", zorder=6)

    # Probability fill
    prob_masked = np.ma.masked_less(prob, PROB_LEVELS[0])
    ax.contourf(
        lons, lats, prob_masked,
        levels=PROB_LEVELS,
        colors=PROB_COLORS,
        alpha=0.80,
        transform=pc,
        zorder=7,
        extend="max",
    )
    ax.contour(
        lons, lats, prob,
        levels=PROB_LEVELS,
        colors=["#00000055"],
        linewidths=0.5,
        transform=pc,
        zorder=8,
    )

    # Lat/lon gridlines
    gl = ax.gridlines(
        crs=pc, draw_labels=True,
        linewidth=0.5, color="#888888", alpha=0.6, linestyle="--",
        x_inline=False, y_inline=False,
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xlocator      = mticker.FixedLocator(x_ticks)
    gl.ylocator      = mticker.FixedLocator(y_ticks)
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 7.5, "color": "#cccccc"}
    gl.ylabel_style  = {"size": 7.5, "color": "#cccccc"}

    # Colorbar
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

    # Header
    run_str   = run_time.strftime("%d %b %Y  %H:%M UTC")
    valid_str = valid_time.strftime("%d %b %Y  %H:%M UTC")
    fxx_str   = f"F{fxx:02d}"
    prod_str  = "6-Hour Maximum Envelope" if is_max else f"Hour {fxx_str}"

    if model == "combined":
        members_str = " + ".join(
            MODEL_CONFIGS[m]["label"] for m in (ensemble_members or ENSEMBLE_MEMBERS)
        )
        model_label = f"Combined Ensemble  ({members_str})"
    else:
        model_label = MODEL_CONFIGS[model]["label_long"]

    title_main = f"{model_label}  |  Tornado Probability  |  {region_label}"
    title_sub  = f"Run: {run_str}     Valid: {valid_str}     {prod_str}"

    ax.set_title(title_main, loc="left",  fontsize=13, fontweight="bold",
                 color="white", pad=6,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#1a1a2e")])
    ax.set_title(title_sub,  loc="right", fontsize=8.5, color="#dddddd", pad=6,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a2e")])

    # Combined: add small per-model legend dots in the lower-right corner
    if model == "combined" and ensemble_members:
        legend_x = 0.99
        legend_y = 0.04
        for i, mdl in enumerate(ensemble_members):
            cfg = MODEL_CONFIGS[mdl]
            ax.plot([], [], "o", color=cfg["color"], markersize=6,
                    label=cfg["label_long"], transform=ax.transAxes)
        leg = ax.legend(loc="lower right", fontsize=7.5, framealpha=0.55,
                        facecolor="#1a1a2e", edgecolor="#444444",
                        labelcolor="white", markerscale=1.0)

    # Bottom caption
    fig.text(
        0.01, 0.022,
        "Ingredients: SBCAPE · 0–3 km SRH · 1-hr Max 0–3 km UH · "
        "STP (CAPE × SRH₁ × BS₀₋₆ × LCL term) · 0–6 km Bulk Shear",
        fontsize=7.5, color="#aaaaaa", ha="left", va="bottom",
    )
    src_note = "NOAA HRRR + RAP + NAM via Herbie" if model == "combined" else \
               f"NOAA {MODEL_CONFIGS[model]['label']} via Herbie"
    fig.text(
        0.99, 0.022,
        f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC  |  {src_note}",
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
    key = (entry["run_time"], entry.get("model", "hrrr"))
    index["runs"] = [
        r for r in index["runs"]
        if not (r["run_time"] == entry["run_time"] and
                r.get("model", "hrrr") == entry.get("model", "hrrr"))
    ]
    index["runs"].insert(0, entry)
    index["runs"] = index["runs"][:48]   # keep up to 48 entries (12 runs × 4 models)
    index["last_updated"] = datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Single-model run
# ---------------------------------------------------------------------------
def run_single_model(run_time, run_key, run_dir, region, n_hours, model):
    """Process all forecast hours for one model and return the index entry."""
    cfg          = MODEL_CONFIGS[model]
    model_label  = cfg["label"]
    fname_prefix = f"tornado_prob_{model}"
    if region == "conus":
        fname_prefix += "_conus"

    print(f"\n{'='*60}")
    print(f"  Model : {cfg['label_long']}")
    print(f"  Region: {'CONUS' if region == 'conus' else 'Northeast US'}")
    print(f"{'='*60}")

    hourly_probs = []
    hourly_maps  = []

    for fxx in range(1, n_hours + 1):
        valid_time = run_time + timedelta(hours=fxx)
        print(f"\n── {model_label} F{fxx:02d}  ({valid_time:%H:%M UTC}) ──")
        try:
            lats, lons, prob_sm = get_model_prob(run_time, fxx, model)
        except Exception as exc:
            print(f"  [ERROR] Skipping {model_label} F{fxx:02d}: {exc}")
            continue

        hourly_probs.append((valid_time, prob_sm, lats, lons))

        fname = f"{fname_prefix}_F{fxx:02d}.png"
        fpath = os.path.join(run_dir, fname)
        make_map(lats, lons, prob_sm, run_time, valid_time,
                 fxx=fxx, is_max=False, output_path=fpath,
                 region=region, model=model)
        hourly_maps.append({
            "fxx":        fxx,
            "valid_time": valid_time.isoformat(),
            "file":       f"{run_key}/{fname}",
        })

    if not hourly_probs:
        print(f"  [ERROR] No hours processed for {model_label} — skipping max map.")
        return None

    # 6-hour max envelope
    print(f"\n── {model_label} 6-hr Maximum Envelope ──")
    all_probs = np.stack([p for _, p, _, _ in hourly_probs], axis=0)
    prob_max  = np.max(all_probs, axis=0)
    lats_r, lons_r = hourly_probs[0][2], hourly_probs[0][3]
    valid_end = hourly_probs[-1][0]

    max_fname = f"{fname_prefix}_MAX.png"
    max_fpath = os.path.join(run_dir, max_fname)
    make_map(lats_r, lons_r, prob_max, run_time, valid_end,
             fxx=n_hours, is_max=True, output_path=max_fpath,
             region=region, model=model)

    return {
        "run_time":      run_time.isoformat(),
        "run_key":       run_key,
        "model":         model,
        "region":        region,
        "n_hours":       n_hours,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "max_prob_map":  f"{run_key}/{max_fname}",
        "hourly_maps":   hourly_maps,
    }


# ---------------------------------------------------------------------------
# Combined-model run
# ---------------------------------------------------------------------------
def run_combined(run_time, run_key, run_dir, region, n_hours):
    """
    Generate per-model maps AND a combined ensemble map for each forecast hour.
    Returns the combined index entry.
    """
    fname_prefix = "tornado_prob_combined"
    if region == "conus":
        fname_prefix += "_conus"

    print(f"\n{'='*60}")
    print(f"  Mode  : Combined Ensemble (HRRR + RAP + NAM)")
    print(f"  Region: {'CONUS' if region == 'conus' else 'Northeast US'}")
    print(f"{'='*60}")

    # -- Phase 1: individual model maps (best-effort) ---------------------
    per_model_probs = {m: [] for m in ENSEMBLE_MEMBERS}   # fxx-indexed native probs

    for model in ENSEMBLE_MEMBERS:
        print(f"\n>>> Individual maps for {MODEL_CONFIGS[model]['label_long']}")
        for fxx in range(1, n_hours + 1):
            valid_time = run_time + timedelta(hours=fxx)
            try:
                lats, lons, prob_sm = get_model_prob(run_time, fxx, model)
                per_model_probs[model].append((fxx, valid_time, lats, lons, prob_sm))
                # Individual model map
                cfg = MODEL_CONFIGS[model]
                fp  = f"tornado_prob_{model}"
                if region == "conus":
                    fp += "_conus"
                fname = f"{fp}_F{fxx:02d}.png"
                fpath = os.path.join(run_dir, fname)
                make_map(lats, lons, prob_sm, run_time, valid_time,
                         fxx=fxx, is_max=False, output_path=fpath,
                         region=region, model=model)
            except Exception as exc:
                print(f"  [WARN] {model.upper()} F{fxx:02d}: {exc}")

        # Individual max-envelope map
        entries = per_model_probs[model]
        if entries:
            all_p    = np.stack([e[4] for e in entries], axis=0)
            prob_max = np.max(all_p, axis=0)
            lats_r   = entries[0][2]
            lons_r   = entries[0][3]
            valid_end = entries[-1][1]
            fp = f"tornado_prob_{model}"
            if region == "conus":
                fp += "_conus"
            max_fname = f"{fp}_MAX.png"
            max_fpath = os.path.join(run_dir, max_fname)
            make_map(lats_r, lons_r, prob_max, run_time, valid_end,
                     fxx=n_hours, is_max=True, output_path=max_fpath,
                     region=region, model=model)

    # -- Phase 2: combined ensemble maps (per hour) ------------------------
    print(f"\n>>> Combined ensemble maps")
    hourly_combined = []
    hourly_maps     = []
    used_members    = [m for m in ENSEMBLE_MEMBERS if per_model_probs[m]]

    common_lats, common_lons = make_common_grid(region)

    for fxx in range(1, n_hours + 1):
        valid_time = run_time + timedelta(hours=fxx)
        print(f"\n── Combined F{fxx:02d}  ({valid_time:%H:%M UTC}) ──")

        stack = []
        for model in used_members:
            # Find the matching fxx entry
            entry = next((e for e in per_model_probs[model] if e[0] == fxx), None)
            if entry is None:
                continue
            _, _, lats_m, lons_m, prob_m = entry
            prob_c = interp_to_grid(lats_m, lons_m, prob_m, common_lats, common_lons)
            stack.append(prob_c)
            print(f"    {MODEL_CONFIGS[model]['label']:6s}  max = {prob_m.max():.1f}%")

        if not stack:
            print(f"  [WARN] No models available for combined F{fxx:02d} — skipping.")
            continue

        prob_ens = np.mean(stack, axis=0).astype(np.float32)
        sigma_c  = RADIUS_KM / COMMON_GRID_KM
        prob_ens = gaussian_filter(prob_ens, sigma=sigma_c)
        prob_ens = np.clip(prob_ens, 0, 95.0).astype(np.float32)

        hourly_combined.append((valid_time, prob_ens, common_lats, common_lons))

        fname = f"{fname_prefix}_F{fxx:02d}.png"
        fpath = os.path.join(run_dir, fname)
        make_map(common_lats, common_lons, prob_ens, run_time, valid_time,
                 fxx=fxx, is_max=False, output_path=fpath,
                 region=region, model="combined",
                 ensemble_members=used_members)
        hourly_maps.append({
            "fxx":        fxx,
            "valid_time": valid_time.isoformat(),
            "file":       f"{run_key}/{fname}",
        })

    if not hourly_combined:
        print("[ERROR] No combined forecast hours processed.")
        return None

    # Combined max-envelope
    print(f"\n── Combined 6-hr Maximum Envelope ──")
    all_probs = np.stack([p for _, p, _, _ in hourly_combined], axis=0)
    prob_max  = np.max(all_probs, axis=0)
    valid_end = hourly_combined[-1][0]

    max_fname = f"{fname_prefix}_MAX.png"
    max_fpath = os.path.join(run_dir, max_fname)
    make_map(common_lats, common_lons, prob_max, run_time, valid_end,
             fxx=n_hours, is_max=True, output_path=max_fpath,
             region=region, model="combined",
             ensemble_members=used_members)

    return {
        "run_time":       run_time.isoformat(),
        "run_key":        run_key,
        "model":          "combined",
        "ensemble":       used_members,
        "region":         region,
        "n_hours":        n_hours,
        "generated_utc":  datetime.utcnow().isoformat() + "Z",
        "max_prob_map":   f"{run_key}/{max_fname}",
        "hourly_maps":    hourly_maps,
    }


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
        description="Generate multi-model tornado probability maps (HRRR / RAP / NAM / Combined)")
    parser.add_argument(
        "--run-time", default="",
        help="Model init time (UTC): 'YYYY-MM-DD HH' or 'YYYY-MM-DD HH:MM'. "
             "Leave blank for the most-recent available run.")
    parser.add_argument(
        "--hours", type=int, default=6,
        help="Forecast hours to generate: 1–6 (default 6).")
    parser.add_argument(
        "--conus", action="store_true",
        help="Generate maps for the full CONUS instead of Northeast US only.")
    parser.add_argument(
        "--model",
        choices=["hrrr", "rap", "nam", "combined"],
        default="combined",
        help="Model to use: hrrr | rap | nam | combined (default: combined).")
    args = parser.parse_args()

    region = "conus" if args.conus else "northeast"

    run_time = parse_run_time(args.run_time)
    if run_time is None:
        now      = datetime.utcnow()
        run_time = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)

    n_hours  = max(1, min(args.hours, 6))
    run_key  = run_time.strftime("%Y%m%d_%H%M")
    run_dir  = os.path.join(OUTPUT_DIR, run_key)
    os.makedirs(run_dir, exist_ok=True)

    region_label = "CONUS" if region == "conus" else "Northeast US"
    print(f"\nTornado Probability Map Generator  —  {args.model.upper()}")
    print(f"Run   : {run_time:%Y-%m-%d %H:%M UTC}")
    print(f"Region: {region_label}")
    print(f"Hours : F01 – F{n_hours:02d}\n")

    index   = load_index()
    entries = []

    if args.model == "combined":
        entry = run_combined(run_time, run_key, run_dir, region, n_hours)
        if entry:
            entries.append(entry)
        # Also individual model runs are handled inside run_combined
    else:
        entry = run_single_model(run_time, run_key, run_dir, region, n_hours, args.model)
        if entry:
            entries.append(entry)

    for entry in entries:
        register_run(index, entry)

    save_index(index)
    print(f"\nIndex updated → {INDEX_FILE}")

    # latest.json for web viewer
    latest_model = args.model if args.model != "combined" else "combined"
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump({
            "run_key":   run_key,
            "run_time":  run_time.isoformat(),
            "model":     latest_model,
        }, f)

    print("Done!\n")


if __name__ == "__main__":
    main()
