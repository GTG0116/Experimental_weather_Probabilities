#!/usr/bin/env python3
"""
Multi-Model Severe Weather Probability Map Generator
=====================================================
Generates 6-hour severe weather probability maps for the Northeast US or CONUS
using HRRR, RRFS, NAM, NSSL-MPAS, or an ensemble combination.

Supported models (--model flag):
  hrrr      : High-Resolution Rapid Refresh (3-km)
  rrfs      : Rapid Refresh Forecast System (3-km; replaces RAP)
  nam       : NAM CONUS Nest (3-km convection-allowing)
  nssl-mpas : NSSL Model for Prediction Across Scales (3-km, experimental)
  combined  : Ensemble average of all available models

Hazard types generated each run:
  tornado   : Tornado probability within 25 miles
  hail      : Large hail (≥1") probability within 25 miles
  wind      : Damaging wind (>60 mph) probability within 25 miles
  lightning : Lightning probability within 25 miles
  severe    : Any severe weather probability within 25 miles

Composite ingredients (GRIB2 SFC fields):
  CAPE:surface                        Surface-Based CAPE
  HLCY:3000-0 m above ground          0-3 km Storm-Relative Helicity
  HLCY:1000-0 m above ground          0-1 km SRH (STP)
  MXUPHL:3000-0 m above ground        1-hr Max 0-3 km Updraft Helicity
  MXUPHL:5000-2000 m above ground     1-hr Max 5-2 km Updraft Helicity (hail)
  VUCSH / VVCSH:0-6000 m              0-6 km Bulk Wind Shear
  HGT:level of adiabatic condensation LCL Height (STP)
  GUST:surface                        Surface Wind Gust (damaging wind)
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
CONUS_EXTENT     = [-125.0, -66.0, 24.0, 50.0]   # [W, E, S, N]
GRID_SPACING_KM  = 3.0
RADIUS_MILES     = 25.0
RADIUS_KM        = RADIUS_MILES * 1.60934

# SPC-style probability levels and colors
PROB_LEVELS = [2, 5, 10, 15, 25, 35, 45, 60]
PROB_COLORS = [
    "#01a101",   #  2%  dark green
    "#79c900",   #  5%  yellow-green
    "#f5f500",   # 10%  yellow
    "#e8a400",   # 15%  orange
    "#fc5d00",   # 25%  dark orange
    "#e30000",   # 35%  red
    "#c000c0",   # 45%  magenta
    "#770077",   # 60%  dark magenta
]

OUTPUT_DIR = os.path.join("docs", "maps")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.json")

# ---------------------------------------------------------------------------
# Hazard metadata — controls map titles, colorbar labels, ingredient text
# ---------------------------------------------------------------------------
HAZARD_INFO = {
    "tornado": {
        "label":       "Tornado",
        "title":       "Tornado Probability",
        "cbar_label":  "Tornado Probability Within 25 Miles",
        "ingredients": (
            "SBCAPE · 0–3 km SRH · 1-hr Max 0–3 km UH · "
            "STP (CAPE × SRH₁ × BS₀₋₆ × LCL) · 0–6 km Bulk Shear"
        ),
    },
    "hail": {
        "label":       "Large Hail",
        "title":       "Large Hail (≥1\") Probability",
        "cbar_label":  "Large Hail (≥1\") Probability Within 25 Miles",
        "ingredients": (
            "SBCAPE · 0–6 km Bulk Shear · 1-hr Max 5–2 km Updraft Helicity · "
            "SHIP-analog composite"
        ),
    },
    "wind": {
        "label":       "Damaging Wind",
        "title":       "Damaging Wind (>60 mph) Probability",
        "cbar_label":  "Damaging Wind (>60 mph) Probability Within 25 Miles",
        "ingredients": (
            "Surface Wind Gust · SBCAPE · 0–6 km Bulk Shear · "
            "Convective Wind Potential"
        ),
    },
    "lightning": {
        "label":       "Lightning",
        "title":       "Lightning Probability",
        "cbar_label":  "Lightning Probability Within 25 Miles",
        "ingredients": (
            "SBCAPE · 0–3 km SRH · Atmospheric Instability Index"
        ),
    },
    "severe": {
        "label":       "Overall Severe",
        "title":       "Overall Severe Weather Probability",
        "cbar_label":  "Any Severe Weather Probability Within 25 Miles",
        "ingredients": (
            "Probabilistic union of Tornado + Large Hail + Damaging Wind"
        ),
    },
}

ALL_HAZARDS = list(HAZARD_INFO.keys())

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
        "color":        "#4e9af1",   # blue
    },
    "rrfs": {
        # Rapid Refresh Forecast System — operational 3-km CONUS CAM
        # replaces the 13-km RAP in this ensemble
        "herbie_model": "rrfs",
        "product":      "conus",     # RRFS deterministic CONUS product
        "label":        "RRFS",
        "label_long":   "RRFS 3-km",
        "grid_km":      3.0,
        "color":        "#f1a44e",   # orange
    },
    "nam": {
        "herbie_model": "nam",
        "product":      "conusnest.hiresf",  # 3-km CONUS convection-allowing nest
        "label":        "NAM",
        "label_long":   "NAM-CONUS 3-km",
        "grid_km":      3.0,
        "color":        "#4ef1a4",   # green
    },
    "nssl-mpas": {
        # NSSL Model for Prediction Across Scales — experimental 3-km CONUS
        # May not be available via standard Herbie backends; fails gracefully.
        "herbie_model": "nssl-mpas",
        "product":      None,
        "label":        "MPAS",
        "label_long":   "NSSL-MPAS 3-km",
        "grid_km":      3.0,
        "color":        "#c87af5",   # purple
    },
}

# Models that form the combined ensemble (tried in order; failures skipped)
ENSEMBLE_MEMBERS = ["hrrr", "rrfs", "nam", "nssl-mpas"]

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
    """
    Download all needed SFC fields for one model / forecast hour.
    Fetches fields for ALL hazard types in one pass.
    """
    cfg = MODEL_CONFIGS[model]
    print(f"  Fetching {cfg['label']} {run_time:%Y-%m-%d %H}z F{fxx:02d} …")
    H = _herbie_instance(run_time, model, fxx)

    # --- Tornado fields ---
    cape_da = fetch(H, r"CAPE:surface",                                  "SBCAPE")
    srh3_da = fetch(H, r"HLCY:3000-0 m above ground",                   "0-3km SRH")
    srh1_da = fetch(H, r"HLCY:1000-0 m above ground",                   "0-1km SRH")
    uh03_da = fetch(H, r"MXUPHL:3000-0 m above ground",                 "0-3km UH")
    shu_da  = fetch(H, r"VUCSH:0-6000 m above ground",                  "BS U")
    shv_da  = fetch(H, r"VVCSH:0-6000 m above ground",                  "BS V")
    lcl_da  = fetch(H, r"HGT:level of adiabatic condensation from sfc", "LCL")

    # --- Additional hail field ---
    uh52_da = fetch(H, r"MXUPHL:5000-2000 m above ground",              "5-2km UH")

    # --- Additional wind field ---
    gust_da = fetch(H, r"GUST:surface",                                  "Sfc Gust")

    if cape_da is None:
        raise RuntimeError(f"Essential field CAPE missing for {model} fxx={fxx}")

    lats = cape_da.latitude.values
    lons = cape_da.longitude.values

    def v(da):
        return da.values.astype(np.float32) if da is not None else None

    cape_np  = v(cape_da)
    srh1_np  = v(srh1_da)
    shu_np   = v(shu_da)
    shv_np   = v(shv_da)
    lcl_np   = v(lcl_da)
    uh03_np  = v(uh03_da)
    uh52_np  = v(uh52_da)
    gust_np  = v(gust_da)

    # Compute Significant Tornado Parameter
    # STP = (SBCAPE/1500) × (SRH₁/150) × (BS₀₋₆/20) × ((2000-LCL)/1000)
    if (cape_np is not None and srh1_np is not None
            and shu_np is not None and shv_np is not None):
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
        "lats": lats,
        "lons": lons,
        "cape": cape_np,
        "srh3": v(srh3_da),
        "uh03": np.abs(uh03_np) if uh03_np is not None else None,
        "uh52": np.abs(uh52_np) if uh52_np is not None else None,
        "bs_u": shu_np,
        "bs_v": shv_np,
        "stp":  stp,
        "gust": gust_np,
    }


# ---------------------------------------------------------------------------
# Probability computation — one function per hazard
# ---------------------------------------------------------------------------
def _bs(fields):
    """0-6 km bulk wind shear magnitude (m/s), or zeros."""
    if fields["bs_u"] is not None and fields["bs_v"] is not None:
        return np.sqrt(fields["bs_u"] ** 2 + fields["bs_v"] ** 2)
    return np.zeros(fields["cape"].shape, np.float32)


def compute_tornado_prob(fields: dict) -> np.ndarray:
    """
    Weighted composite tornado probability index → 0–95%.

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
    bs   = _bs(fields)

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
    uh_gate = np.clip(uh03 / 50.0, 0.0, 1.0) ** 0.6
    if fields["uh03"] is None:
        uh_gate = np.ones(shape, np.float32)
    composite = composite * uh_gate

    prob = 100.0 * (1.0 - np.exp(-0.9 * composite ** 1.6))
    return np.clip(prob, 0, 95.0).astype(np.float32)


def compute_hail_prob(fields: dict) -> np.ndarray:
    """
    Large hail (≥1") probability index → 0–95%.

    Ingredients:
      - 5-2 km Updraft Helicity: primary indicator of supercell hail potential
      - SBCAPE: instability driving updraft strength
      - 0-6 km Bulk Shear: storm organization / longevity

    Based on a SHIP-analog approach:
      composite = 0.45*UH52 + 0.30*CAPE + 0.25*shear
    Thresholds: UH52≥100 m²/s², CAPE≥2000 J/kg, BS≥25 m/s → high hail prob.
    """
    shape = fields["cape"].shape
    z     = np.zeros(shape, np.float32)

    uh52 = fields["uh52"] if fields["uh52"] is not None else z
    cape = np.clip(fields["cape"], 0, None)
    bs   = _bs(fields)

    uh52_n = np.clip(uh52 / 100.0, 0, 3.0)
    cape_n = np.clip(cape / 2000.0, 0, 2.5)
    bs_n   = np.clip(bs   / 25.0,  0, 2.5)

    composite = (
        0.45 * uh52_n +
        0.30 * cape_n +
        0.25 * bs_n
    )

    # UH gate: hail requires a significant rotating updraft
    if fields["uh52"] is not None:
        uh_gate = np.clip(uh52 / 75.0, 0.0, 1.0) ** 0.5
    else:
        # Fall back to 0-3km UH if 5-2km not available
        uh03 = fields["uh03"] if fields["uh03"] is not None else z
        uh_gate = np.clip(uh03 / 75.0, 0.0, 1.0) ** 0.5

    composite = composite * uh_gate

    prob = 100.0 * (1.0 - np.exp(-0.70 * composite ** 1.5))
    return np.clip(prob, 0, 95.0).astype(np.float32)


def compute_wind_prob(fields: dict) -> np.ndarray:
    """
    Damaging wind gust (>60 mph / 26.8 m/s) probability index → 0–95%.

    Ingredients:
      - Surface wind gust (primary when available; 60 mph = 26.8 m/s threshold)
      - SBCAPE + 0-6 km shear (convective wind potential when gust unavailable)

    With gust data:    0.55*gust + 0.25*CAPE + 0.20*shear
    Without gust data: 0.50*CAPE + 0.50*shear  (reduced confidence)
    """
    shape = fields["cape"].shape
    z     = np.zeros(shape, np.float32)

    gust = fields["gust"]
    cape = np.clip(fields["cape"], 0, None)
    bs   = _bs(fields)

    cape_n = np.clip(cape / 1500.0, 0, 2.0)
    bs_n   = np.clip(bs   / 20.0,  0, 2.5)

    if gust is not None:
        gust_n    = np.clip(gust / 26.8, 0, 3.0)   # 1.0 ≡ 60 mph
        composite = 0.55 * gust_n + 0.25 * cape_n + 0.20 * bs_n
    else:
        composite = 0.50 * cape_n + 0.50 * bs_n

    prob = 100.0 * (1.0 - np.exp(-0.85 * composite ** 1.7))
    return np.clip(prob, 0, 95.0).astype(np.float32)


def compute_lightning_prob(fields: dict) -> np.ndarray:
    """
    Lightning probability index → 0–95%.

    Ingredients:
      - SBCAPE: primary driver (Price-Rind: flash rate ∝ updraft / cloud-top proxy)
      - 0-3 km SRH: organized convection → higher and more persistent flash rates

    Formula (after Price & Rind 1992 proxy):
      lightning_idx = 0.70 * (CAPE/500)^0.7 + 0.30 * (SRH3/200)^0.5
    Near-zero below CAPE ~100 J/kg; saturates above CAPE ~3000 J/kg.
    """
    cape = np.clip(fields["cape"], 0, None)
    srh3 = (np.clip(fields["srh3"], 0, None)
            if fields["srh3"] is not None
            else np.zeros(fields["cape"].shape, np.float32))

    cape_n = np.clip(cape / 500.0, 0, 4.0) ** 0.7
    srh_n  = np.clip(srh3 / 200.0, 0, 2.0) ** 0.5

    composite = 0.70 * cape_n + 0.30 * srh_n

    prob = 100.0 * (1.0 - np.exp(-0.60 * composite ** 1.4))
    return np.clip(prob, 0, 95.0).astype(np.float32)


def compute_severe_prob(
    prob_tornado: np.ndarray,
    prob_hail:    np.ndarray,
    prob_wind:    np.ndarray,
) -> np.ndarray:
    """
    Overall severe weather probability — probabilistic union of
    tornado + large hail + damaging wind.

    P(any severe) = 1 - P(no tornado) × P(no hail) × P(no wind)
    """
    p_none = (
        (1.0 - prob_tornado / 100.0) *
        (1.0 - prob_hail    / 100.0) *
        (1.0 - prob_wind    / 100.0)
    )
    prob = 100.0 * (1.0 - p_none)
    return np.clip(prob, 0, 95.0).astype(np.float32)


def smooth(prob: np.ndarray, model: str = "hrrr") -> np.ndarray:
    """Gaussian smoothing representing 25-mile positional uncertainty."""
    grid_km = MODEL_CONFIGS[model]["grid_km"]
    sigma   = RADIUS_KM / grid_km
    return gaussian_filter(prob, sigma=sigma)


def compute_all_probs(fields: dict, model: str) -> dict:
    """
    Compute and smooth all five hazard probability grids from a fields dict.
    Returns a dict keyed by hazard name.
    """
    tor   = compute_tornado_prob(fields)
    hail  = compute_hail_prob(fields)
    wind  = compute_wind_prob(fields)
    light = compute_lightning_prob(fields)
    # Compute severe from raw (unsmoothed) component probs, then smooth together
    sev   = compute_severe_prob(tor, hail, wind)

    return {
        "tornado":   smooth(tor,   model),
        "hail":      smooth(hail,  model),
        "wind":      smooth(wind,  model),
        "lightning": smooth(light, model),
        "severe":    smooth(sev,   model),
    }


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
    Interpolate src_data onto the regular target lat/lon grid.
    Fills outside convex hull with 0.0.
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


def get_model_probs(run_time: datetime, fxx: int, model: str) -> tuple:
    """
    Fetch fields, compute all hazard probabilities, smooth.
    Returns (lats, lons, probs_dict) or raises on failure.
    """
    fields = get_fields(run_time, fxx, model)
    probs  = compute_all_probs(fields, model)
    return fields["lats"], fields["lons"], probs


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------
def make_map(
    lats, lons, prob,
    run_time:   datetime,
    valid_time: datetime,
    fxx:        int,
    is_max:     bool,
    output_path: str,
    region:     str = "conus",
    model:      str = "hrrr",
    hazard:     str = "tornado",
    ensemble_members: list = None,
):
    hinfo = HAZARD_INFO[hazard]

    if region == "conus":
        W, E, S, N = CONUS_EXTENT
        proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5,
                                     standard_parallels=(33, 45))
        figsize      = (22, 13)
        x_ticks      = list(range(-125, -65, 5))
        y_ticks      = list(range(24, 51, 4))
        region_label = "CONUS"
    else:
        W, E, S, N = NORTHEAST_EXTENT
        proj = ccrs.LambertConformal(central_longitude=-73.5, central_latitude=42.5,
                                     standard_parallels=(33, 45))
        figsize      = (16, 11)
        x_ticks      = list(range(-82, -63, 2))
        y_ticks      = list(range(36, 50, 2))
        region_label = "Northeast US"

    pc = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("#1a1a2e")

    ax = fig.add_axes([0.01, 0.10, 0.98, 0.84], projection=proj)
    ax.set_extent([W, E, S, N], crs=pc)
    ax.set_facecolor("#1c3557")

    # Background features
    LAND    = cfeature.NaturalEarthFeature("physical", "land",          "50m")
    OCEAN   = cfeature.NaturalEarthFeature("physical", "ocean",         "50m")
    LAKES   = cfeature.NaturalEarthFeature("physical", "lakes",         "50m")
    RIVERS  = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m")
    STATES  = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m")
    COUNTIES = cfeature.NaturalEarthFeature("cultural", "admin_2_counties",              "10m")
    BORDERS = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land",   "50m")
    COAST   = cfeature.NaturalEarthFeature("physical", "coastline",     "50m")

    ax.add_feature(OCEAN,    facecolor="#1c3557",   zorder=0)
    ax.add_feature(LAND,     facecolor="#f0ede6",   zorder=1)
    ax.add_feature(LAKES,    facecolor="#1c3557",   zorder=2)
    ax.add_feature(RIVERS,   edgecolor="#7baed4", linewidth=0.3,  facecolor="none", zorder=3)
    ax.add_feature(COUNTIES, edgecolor="#bbbbbb", linewidth=0.25, facecolor="none", zorder=4)
    ax.add_feature(STATES,   edgecolor="#555555", linewidth=0.8,  facecolor="none", zorder=5)
    ax.add_feature(BORDERS,  edgecolor="#222222", linewidth=1.2,  facecolor="none", zorder=6)
    ax.add_feature(COAST,    edgecolor="#222222", linewidth=0.7,  facecolor="none", zorder=6)

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
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator(x_ticks)
    gl.ylocator     = mticker.FixedLocator(y_ticks)
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 7.5, "color": "#cccccc"}
    gl.ylabel_style = {"size": 7.5, "color": "#cccccc"}

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
    cbar_ax.set_xlabel(hinfo["cbar_label"], fontsize=9.5, color="white", labelpad=5)

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

    title_main = f"{model_label}  |  {hinfo['title']}  |  {region_label}"
    title_sub  = f"Run: {run_str}     Valid: {valid_str}     {prod_str}"

    ax.set_title(title_main, loc="left",  fontsize=13, fontweight="bold",
                 color="white", pad=6,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#1a1a2e")])
    ax.set_title(title_sub,  loc="right", fontsize=8.5, color="#dddddd", pad=6,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a2e")])

    # Combined: per-model legend dots
    if model == "combined" and ensemble_members:
        for mdl in ensemble_members:
            cfg = MODEL_CONFIGS[mdl]
            ax.plot([], [], "o", color=cfg["color"], markersize=6,
                    label=cfg["label_long"], transform=ax.transAxes)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.55,
                  facecolor="#1a1a2e", edgecolor="#444444",
                  labelcolor="white", markerscale=1.0)

    # Bottom caption
    fig.text(
        0.01, 0.022,
        f"Ingredients: {hinfo['ingredients']}",
        fontsize=7.5, color="#aaaaaa", ha="left", va="bottom",
    )
    if model == "combined":
        src_note = "NOAA HRRR + RRFS + NAM + NSSL-MPAS via Herbie"
    else:
        src_note = f"NOAA {MODEL_CONFIGS[model]['label']} via Herbie"
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
# File-name helper
# ---------------------------------------------------------------------------
def _fname_prefix(hazard: str, model: str, region: str) -> str:
    """e.g. 'tornado_prob_hrrr_conus' or 'hail_prob_combined'"""
    p = f"{hazard}_prob_{model}"
    if region == "conus":
        p += "_conus"
    return p


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
    index["runs"] = index["runs"][:60]   # keep up to 60 entries
    index["last_updated"] = datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Single-model run
# ---------------------------------------------------------------------------
def run_single_model(run_time, run_key, run_dir, region, n_hours, model):
    """
    Process all forecast hours for one model.
    Generates maps for all five hazard types.
    Returns the index entry dict.
    """
    cfg         = MODEL_CONFIGS[model]
    model_label = cfg["label"]

    print(f"\n{'='*60}")
    print(f"  Model : {cfg['label_long']}")
    print(f"  Region: {'CONUS' if region == 'conus' else 'Northeast US'}")
    print(f"  Hazards: {', '.join(ALL_HAZARDS)}")
    print(f"{'='*60}")

    # hourly_data[hazard] = list of (valid_time, prob, lats, lons)
    hourly_data = {h: [] for h in ALL_HAZARDS}
    hourly_maps = []   # index entries (tornado maps for backward compat)

    for fxx in range(1, n_hours + 1):
        valid_time = run_time + timedelta(hours=fxx)
        print(f"\n── {model_label} F{fxx:02d}  ({valid_time:%H:%M UTC}) ──")
        try:
            lats, lons, probs = get_model_probs(run_time, fxx, model)
        except Exception as exc:
            print(f"  [ERROR] Skipping {model_label} F{fxx:02d}: {exc}")
            continue

        for hazard in ALL_HAZARDS:
            prob_sm = probs[hazard]
            hourly_data[hazard].append((valid_time, prob_sm, lats, lons))
            fname = f"{_fname_prefix(hazard, model, region)}_F{fxx:02d}.png"
            fpath = os.path.join(run_dir, fname)
            make_map(lats, lons, prob_sm, run_time, valid_time,
                     fxx=fxx, is_max=False, output_path=fpath,
                     region=region, model=model, hazard=hazard)
            if hazard == "tornado":
                hourly_maps.append({
                    "fxx":        fxx,
                    "valid_time": valid_time.isoformat(),
                    "file":       f"{run_key}/{fname}",
                })
            print(f"    {hazard:10s} max={prob_sm.max():.1f}%")

    if not hourly_data["tornado"]:
        print(f"  [ERROR] No hours processed for {model_label} — skipping max maps.")
        return None

    # Per-hazard 6-hr max envelope
    for hazard in ALL_HAZARDS:
        entries = hourly_data[hazard]
        if not entries:
            continue
        print(f"\n── {model_label} 6-hr Max  [{hazard}] ──")
        all_p     = np.stack([p for _, p, _, _ in entries], axis=0)
        prob_max  = np.max(all_p, axis=0)
        lats_r    = entries[0][2]
        lons_r    = entries[0][3]
        valid_end = entries[-1][0]
        max_fname = f"{_fname_prefix(hazard, model, region)}_MAX.png"
        max_fpath = os.path.join(run_dir, max_fname)
        make_map(lats_r, lons_r, prob_max, run_time, valid_end,
                 fxx=n_hours, is_max=True, output_path=max_fpath,
                 region=region, model=model, hazard=hazard)

    # Return entry using tornado max map for backward compat
    tor_entries = hourly_data["tornado"]
    lats_r, lons_r = tor_entries[0][2], tor_entries[0][3]
    max_fname_tor  = f"{_fname_prefix('tornado', model, region)}_MAX.png"
    return {
        "run_time":      run_time.isoformat(),
        "run_key":       run_key,
        "model":         model,
        "region":        region,
        "n_hours":       n_hours,
        "hazards":       ALL_HAZARDS,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "max_prob_map":  f"{run_key}/{max_fname_tor}",
        "hourly_maps":   hourly_maps,
    }


# ---------------------------------------------------------------------------
# Combined-model run
# ---------------------------------------------------------------------------
def run_combined(run_time, run_key, run_dir, region, n_hours):
    """
    Generate per-model maps AND combined ensemble maps for all hazard types.
    Returns the combined index entry.
    """
    print(f"\n{'='*60}")
    print(f"  Mode  : Combined Ensemble (HRRR + RRFS + NAM + NSSL-MPAS)")
    print(f"  Region: {'CONUS' if region == 'conus' else 'Northeast US'}")
    print(f"  Hazards: {', '.join(ALL_HAZARDS)}")
    print(f"{'='*60}")

    # Phase 1: per-model individual maps & probability storage
    # per_model_probs[model] = list of (fxx, valid_time, lats, lons, probs_dict)
    per_model_probs = {m: [] for m in ENSEMBLE_MEMBERS}

    for model in ENSEMBLE_MEMBERS:
        print(f"\n>>> Individual maps for {MODEL_CONFIGS[model]['label_long']}")
        for fxx in range(1, n_hours + 1):
            valid_time = run_time + timedelta(hours=fxx)
            try:
                lats, lons, probs = get_model_probs(run_time, fxx, model)
                per_model_probs[model].append((fxx, valid_time, lats, lons, probs))

                for hazard in ALL_HAZARDS:
                    fname = f"{_fname_prefix(hazard, model, region)}_F{fxx:02d}.png"
                    fpath = os.path.join(run_dir, fname)
                    make_map(lats, lons, probs[hazard], run_time, valid_time,
                             fxx=fxx, is_max=False, output_path=fpath,
                             region=region, model=model, hazard=hazard)
                print(f"  F{fxx:02d} tornado max={probs['tornado'].max():.1f}%  "
                      f"hail={probs['hail'].max():.1f}%  "
                      f"wind={probs['wind'].max():.1f}%  "
                      f"ltng={probs['lightning'].max():.1f}%  "
                      f"sev={probs['severe'].max():.1f}%")
            except Exception as exc:
                print(f"  [WARN] {model.upper()} F{fxx:02d}: {exc}")

        # Individual model max-envelope maps (all hazards)
        entries = per_model_probs[model]
        if entries:
            lats_r    = entries[0][2]
            lons_r    = entries[0][3]
            valid_end = entries[-1][1]
            for hazard in ALL_HAZARDS:
                all_p    = np.stack([e[4][hazard] for e in entries], axis=0)
                prob_max = np.max(all_p, axis=0)
                max_fname = f"{_fname_prefix(hazard, model, region)}_MAX.png"
                max_fpath = os.path.join(run_dir, max_fname)
                make_map(lats_r, lons_r, prob_max, run_time, valid_end,
                         fxx=n_hours, is_max=True, output_path=max_fpath,
                         region=region, model=model, hazard=hazard)

    # Phase 2: combined ensemble maps for all hazard types
    print(f"\n>>> Combined ensemble maps")
    used_members     = [m for m in ENSEMBLE_MEMBERS if per_model_probs[m]]
    common_lats, common_lons = make_common_grid(region)
    sigma_c          = RADIUS_KM / COMMON_GRID_KM

    hourly_combined  = []   # (valid_time, combined_probs_dict)
    hourly_maps      = []   # index entries (tornado)

    for fxx in range(1, n_hours + 1):
        valid_time = run_time + timedelta(hours=fxx)
        print(f"\n── Combined F{fxx:02d}  ({valid_time:%H:%M UTC}) ──")

        # For each hazard, stack model probs on common grid
        hazard_stacks = {h: [] for h in ALL_HAZARDS}
        for model in used_members:
            entry = next((e for e in per_model_probs[model] if e[0] == fxx), None)
            if entry is None:
                continue
            _, _, lats_m, lons_m, probs_m = entry
            for hazard in ALL_HAZARDS:
                prob_c = interp_to_grid(lats_m, lons_m, probs_m[hazard],
                                        common_lats, common_lons)
                hazard_stacks[hazard].append(prob_c)
            print(f"    {MODEL_CONFIGS[model]['label']:6s}  "
                  f"tor={probs_m['tornado'].max():.1f}%  "
                  f"hail={probs_m['hail'].max():.1f}%")

        n_in_stack = len(hazard_stacks["tornado"])
        if n_in_stack == 0:
            print(f"  [WARN] No models available for combined F{fxx:02d} — skipping.")
            continue

        combined_probs = {}
        for hazard in ALL_HAZARDS:
            if not hazard_stacks[hazard]:
                combined_probs[hazard] = np.zeros(common_lats.shape, np.float32)
                continue
            prob_ens = np.mean(hazard_stacks[hazard], axis=0).astype(np.float32)
            prob_ens = gaussian_filter(prob_ens, sigma=sigma_c)
            combined_probs[hazard] = np.clip(prob_ens, 0, 95.0).astype(np.float32)

        hourly_combined.append((valid_time, combined_probs))

        for hazard in ALL_HAZARDS:
            fname = f"{_fname_prefix(hazard, 'combined', region)}_F{fxx:02d}.png"
            fpath = os.path.join(run_dir, fname)
            make_map(common_lats, common_lons, combined_probs[hazard],
                     run_time, valid_time,
                     fxx=fxx, is_max=False, output_path=fpath,
                     region=region, model="combined", hazard=hazard,
                     ensemble_members=used_members)
            if hazard == "tornado":
                hourly_maps.append({
                    "fxx":        fxx,
                    "valid_time": valid_time.isoformat(),
                    "file":       f"{run_key}/{fname}",
                })

    if not hourly_combined:
        print("[ERROR] No combined forecast hours processed.")
        return None

    # Combined max-envelope for all hazards
    print(f"\n── Combined 6-hr Maximum Envelope ──")
    valid_end = hourly_combined[-1][0]
    for hazard in ALL_HAZARDS:
        all_p    = np.stack([cp[hazard] for _, cp in hourly_combined], axis=0)
        prob_max = np.max(all_p, axis=0)
        max_fname = f"{_fname_prefix(hazard, 'combined', region)}_MAX.png"
        max_fpath = os.path.join(run_dir, max_fname)
        make_map(common_lats, common_lons, prob_max, run_time, valid_end,
                 fxx=n_hours, is_max=True, output_path=max_fpath,
                 region=region, model="combined", hazard=hazard,
                 ensemble_members=used_members)

    max_fname_tor = f"{_fname_prefix('tornado', 'combined', region)}_MAX.png"
    return {
        "run_time":      run_time.isoformat(),
        "run_key":       run_key,
        "model":         "combined",
        "ensemble":      used_members,
        "region":        region,
        "n_hours":       n_hours,
        "hazards":       ALL_HAZARDS,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "max_prob_map":  f"{run_key}/{max_fname_tor}",
        "hourly_maps":   hourly_maps,
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
        description=(
            "Generate multi-model severe weather probability maps "
            "(HRRR / RRFS / NAM / NSSL-MPAS / Combined). "
            "Each run produces maps for: tornado, large hail, "
            "damaging wind (>60 mph), lightning, and overall severe."
        )
    )
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
        choices=["hrrr", "rrfs", "nam", "nssl-mpas", "combined"],
        default="combined",
        help="Model to use: hrrr | rrfs | nam | nssl-mpas | combined (default: combined).")
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
    print(f"\nSevere Weather Probability Map Generator  —  {args.model.upper()}")
    print(f"Run    : {run_time:%Y-%m-%d %H:%M UTC}")
    print(f"Region : {region_label}")
    print(f"Hours  : F01 – F{n_hours:02d}")
    print(f"Hazards: {', '.join(ALL_HAZARDS)}\n")

    index   = load_index()
    entries = []

    if args.model == "combined":
        entry = run_combined(run_time, run_key, run_dir, region, n_hours)
        if entry:
            entries.append(entry)
    else:
        entry = run_single_model(run_time, run_key, run_dir, region, n_hours, args.model)
        if entry:
            entries.append(entry)

    for entry in entries:
        register_run(index, entry)

    save_index(index)
    print(f"\nIndex updated → {INDEX_FILE}")

    latest_model = args.model
    with open(os.path.join("docs", "latest.json"), "w") as f:
        json.dump({
            "run_key":   run_key,
            "run_time":  run_time.isoformat(),
            "model":     latest_model,
            "hazards":   ALL_HAZARDS,
        }, f)

    print("Done!\n")


if __name__ == "__main__":
    main()
