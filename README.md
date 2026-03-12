# HRRR Tornado Probability Map — Northeast US

Experimental 6-hour tornado probability maps generated from NOAA's
**High-Resolution Rapid Refresh (HRRR)** model for the Northeast United States.

## What It Shows

Each map displays the estimated probability of a tornado occurring
**within 25 statute miles** of any given point, valid for a specific
HRRR forecast hour (F01 – F06).  A **6-hour maximum envelope** map
shows the worst-case probability across the entire period.

## Composite Index

Five HRRR fields are combined into a weighted probability index:

| Ingredient | HRRR Variable | Weight | Threshold |
|---|---|---|---|
| 0–3 km Updraft Helicity | `MXUPHL:3000-0 m above ground` | **38%** | 75 m² s⁻² |
| Significant Tornado Parameter (computed) | `CAPE × SRH₁ × BS₀₋₆` | **28%** | 1.0 |
| 0–3 km Storm-Relative Helicity | `HLCY:3000-0 m above ground` | **18%** | 250 m² s⁻² |
| Surface-Based CAPE | `CAPE:surface` | **10%** | 1 500 J kg⁻¹ |
| 0–6 km Bulk Wind Shear | `VUCSH/VVCSH:0-6000 m above ground` | **6%** | 20 m s⁻¹ (≈ 39 kt) |

An **UH gate** (non-linear multiplicative factor) suppresses probability
in areas with little or no mesocyclone rotation — the primary discriminator
for tornadic vs. non-tornadic supercells.

Raw probability values are convolved with a Gaussian kernel sized to a
~25-mile radius to produce smooth, positional-uncertainty-aware fields.

## Viewing the Maps

**GitHub Pages viewer:**
`https://<your-github-username>.github.io/<repo-name>/`

The viewer shows:
- Latest HRRR run drop-down
- Tab selector for each forecast hour (F01–F06) + 6-hour maximum
- Probability legend
- Map metadata (run time, valid time)

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Generate maps for the latest available HRRR run
python generate_tornado_map.py

# Generate for a specific run and only 3 hours
python generate_tornado_map.py --run-time "2025-05-15 18" --hours 3
```

Maps are saved under `docs/maps/<YYYYMMDD_HHMM>/`.
`docs/maps/index.json` is updated with metadata for the web viewer.

## Automation

A **GitHub Actions** workflow (`.github/workflows/update_maps.yml`) runs
automatically at :05 past every hour to pull the latest HRRR data and
regenerate maps.  It commits the new PNGs + JSON back to `main` and
redeploys to GitHub Pages.

You can also trigger it manually from the **Actions** tab with an optional
specific run time and hour count.

## Repository Layout

```
├── generate_tornado_map.py     # Main Python script
├── requirements.txt            # Python dependencies
├── docs/
│   ├── index.html              # GitHub Pages viewer (SPA)
│   ├── latest.json             # Pointer to most-recent run
│   └── maps/
│       ├── index.json          # Run metadata index
│       └── <YYYYMMDD_HHMM>/
│               ├── tornado_prob_F01.png
│               ├── tornado_prob_F02.png
│               │   …
│               └── tornado_prob_MAX.png
└── .github/
    └── workflows/
        └── update_maps.yml     # Hourly automation
```

## Enabling GitHub Pages

1. Go to **Settings → Pages** in your repository.
2. Under **Source**, select **GitHub Actions**.
3. The workflow will deploy automatically after the first successful run.

## Disclaimer

> **Experimental product — not for operational use.**
> HRRR-based tornado probabilities carry significant uncertainty,
> especially beyond 3 forecast hours.  Always consult official
> NWS / SPC products for severe weather decisions.

Data source: [NOAA HRRR (AWS)](https://registry.opendata.aws/noaa-hrrr-pds/)
Python tooling: [Herbie](https://herbie.readthedocs.io/) · Cartopy · Matplotlib · SciPy
