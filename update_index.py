"""
Update the GitHub Pages index.html with links to all maps in maps/.
Run this after generate_map.py so the new map appears in the archive.
"""

import re
from datetime import datetime
from pathlib import Path

MAPS_DIR = Path("maps")
INDEX_PATH = Path("index.html")

# Collect all tornado-probability HTML files
map_files = sorted(MAPS_DIR.glob("tornado_prob_*.html"), reverse=True)

# Build table rows -----------------------------------------------------------
rows = []
for mf in map_files:
    m = re.search(r"tornado_prob_(\d{8})_(\d{4})\.html", mf.name)
    if not m:
        continue
    date_str, time_str = m.group(1), m.group(2)
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    display = dt.strftime("%Y-%m-%d %H:%M UTC")

    geojsons = sorted(MAPS_DIR.glob("tornado_prob_{d}_{t}_fxx*.geojson".format(d=date_str, t=time_str)))
    gjson_links = " ".join(
        '<a href="maps/{name}" title="GeoJSON fxx{n:02d}" class="gj-link">+{n:02d}h</a>'.format(
            name=g.name, n=i + 1
        )
        for i, g in enumerate(geojsons)
    )
    if not gjson_links:
        gjson_links = "<span style='color:#999'>&#x2014;</span>"

    rows.append(
        "    <tr>\n"
        '      <td><a href="maps/{fn}" class="map-link">{disp}</a></td>\n'
        "      <td>{gj}</td>\n"
        '      <td><a href="maps/{fn}" class="view-btn">&#9654; View Map</a></td>\n'
        "    </tr>".format(fn=mf.name, disp=display, gj=gjson_links)
    )

if rows:
    rows_html = "\n".join(rows)
else:
    rows_html = (
        "    <tr><td colspan='3' style='text-align:center;color:#999;'>"
        "No maps generated yet.</td></tr>"
    )

generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# Latest map block -----------------------------------------------------------
if map_files:
    latest_name = map_files[0].name
    m0 = re.search(r"tornado_prob_(\d{8})_(\d{4})\.html", latest_name)
    latest_run = ""
    if m0:
        latest_dt = datetime.strptime(m0.group(1) + m0.group(2), "%Y%m%d%H%M")
        latest_run = latest_dt.strftime("%Y-%m-%d %H:%M UTC")
    latest_iframe = (
        '<iframe src="maps/{fn}" title="Latest tornado probability map" '
        'loading="lazy"></iframe>'.format(fn=latest_name)
    )
    latest_footer_left = (
        "Run: {run} &nbsp;&middot;&nbsp; "
        "Use the layer control (top-left of map) to toggle forecast hours.".format(run=latest_run)
    )
    latest_footer_right = (
        '<a href="maps/{fn}" class="open-btn" target="_blank">Open Full Screen &#8599;</a>'.format(
            fn=latest_name
        )
    )
else:
    latest_iframe = (
        "<div style='padding:40px;text-align:center;color:var(--muted)'>"
        "No map generated yet. Run the GitHub Action to generate one.</div>"
    )
    latest_footer_left = ""
    latest_footer_right = ""

# Build HTML -----------------------------------------------------------------
# Use a list of string parts to avoid any .format() / f-string issues with CSS
parts = []
parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HRRR Tornado Probability Maps \u2013 Northeast US</title>
  <style>
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --accent: #58a6ff; --text: #c9d1d9; --muted: #8b949e;
      --danger: #f85149; --warn: #e3b341; --ok: #3fb950;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      line-height: 1.6;
    }
    .hero {
      background: linear-gradient(135deg, #1a1f35 0%, #0d1117 60%);
      border-bottom: 1px solid var(--border);
      padding: 48px 24px 36px; text-align: center;
    }
    .hero h1 { font-size: clamp(1.6rem,4vw,2.4rem); font-weight:700; letter-spacing:-0.5px; margin-bottom:10px; }
    .hero h1 span { color: var(--danger); }
    .hero p { color:var(--muted); max-width:640px; margin:0 auto 20px; font-size:0.95rem; }
    .badge {
      display:inline-block; background:#21262d; border:1px solid var(--border);
      border-radius:20px; padding:4px 12px; font-size:0.78rem; color:var(--muted); margin:3px;
    }
    .badge b { color:var(--text); }
    .latest-section { max-width:1000px; margin:32px auto 0; padding:0 24px; }
    .section-title {
      font-size:1.1rem; font-weight:600; color:var(--accent);
      margin-bottom:14px; display:flex; align-items:center; gap:8px;
    }
    .section-title::before {
      content:""; display:inline-block; width:4px; height:18px;
      background:var(--accent); border-radius:2px;
    }
    .latest-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
    .latest-card iframe { width:100%; height:520px; border:none; display:block; }
    .latest-card .card-footer {
      padding:12px 16px; border-top:1px solid var(--border);
      font-size:0.82rem; color:var(--muted);
      display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
    }
    .open-btn {
      background:var(--accent); color:#0d1117; font-weight:600;
      padding:6px 16px; border-radius:6px; text-decoration:none;
      font-size:0.83rem; transition:opacity 0.15s;
    }
    .open-btn:hover { opacity:0.85; }
    .archive-section { max-width:1000px; margin:32px auto 48px; padding:0 24px; }
    table { width:100%; border-collapse:collapse; font-size:0.88rem; }
    thead th {
      background:#21262d; color:var(--muted); font-weight:600;
      text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em;
      padding:10px 14px; border-bottom:1px solid var(--border); text-align:left;
    }
    tbody tr { border-bottom:1px solid var(--border); transition:background 0.1s; }
    tbody tr:hover { background:#21262d; }
    tbody td { padding:10px 14px; vertical-align:middle; }
    .map-link { color:var(--accent); text-decoration:none; font-weight:500; }
    .map-link:hover { text-decoration:underline; }
    .gj-link {
      display:inline-block; background:#21262d; color:var(--muted);
      border:1px solid var(--border); border-radius:4px; padding:2px 7px;
      margin:1px; text-decoration:none; font-size:0.75rem;
      transition:border-color 0.15s,color 0.15s;
    }
    .gj-link:hover { border-color:var(--accent); color:var(--accent); }
    .view-btn {
      background:transparent; border:1px solid var(--ok); color:var(--ok);
      padding:4px 12px; border-radius:5px; text-decoration:none;
      font-size:0.8rem; transition:background 0.15s;
    }
    .view-btn:hover { background:rgba(63,185,80,0.12); }
    .method-section { max-width:1000px; margin:0 auto 48px; padding:0 24px; }
    .method-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-top:14px; }
    .method-card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
    .method-card h3 { font-size:0.9rem; font-weight:700; margin-bottom:6px; color:var(--warn); }
    .method-card p { font-size:0.8rem; color:var(--muted); line-height:1.5; }
    footer { border-top:1px solid var(--border); padding:20px 24px; text-align:center; font-size:0.78rem; color:var(--muted); }
  </style>
</head>
<body>
""")

parts.append("""<div class="hero">
  <h1>HRRR <span>Tornado Probability</span> Maps</h1>
  <p>Automated 6-hour outlook for the <strong>Northeast United States</strong>,
     updated every 6 hours from the NOAA High-Resolution Rapid Refresh (HRRR) model.</p>
  <span class="badge"><b>Model:</b> HRRR 3-km</span>
  <span class="badge"><b>Domain:</b> Northeast US</span>
  <span class="badge"><b>Horizon:</b> +01 &rarr; +06 h</span>
  <span class="badge"><b>Radius:</b> 25-mile</span>
  <span class="badge"><b>Updated:</b> """ + generated_at + """</span>
</div>
""")

parts.append("""<div class="latest-section">
  <div class="section-title">Latest Map</div>
  <div class="latest-card">
    """ + latest_iframe + """
    <div class="card-footer">
      <span>""" + latest_footer_left + """</span>
      """ + latest_footer_right + """
    </div>
  </div>
</div>
""")

parts.append("""<div class="method-section" style="margin-top:32px;">
  <div class="section-title">How It Works</div>
  <div class="method-grid">
    <div class="method-card">
      <h3>CAPE (20%)</h3>
      <p>Convective Available Potential Energy measures atmospheric instability.
         Higher CAPE &rarr; more energetic updrafts. Normalised at 3000 J kg&sup1;.</p>
    </div>
    <div class="method-card">
      <h3>0&ndash;6 km Bulk Shear (20%)</h3>
      <p>Vector difference between surface and 6&nbsp;km AGL winds. Strong deep-layer
         shear organises supercell structure. Normalised at 40 m s&sup1;.</p>
    </div>
    <div class="method-card">
      <h3>Sig. Tornado Param. (30%)</h3>
      <p>STP combines CAPE, shear, LCL height, and storm-relative helicity into a
         single supercell-tornado metric. Normalised at 4.0.</p>
    </div>
    <div class="method-card">
      <h3>0&ndash;3 km UH Max (30%)</h3>
      <p>1-hour maximum 0&ndash;3&nbsp;km AGL Updraft Helicity tracks rotating updrafts
         directly &mdash; strongest predictor of tornado occurrence. Normalised at 200 m&sup2; s&sup2;.</p>
    </div>
  </div>
  <p style="margin-top:12px;font-size:0.8rem;color:var(--muted);">
    The four normalised fields are combined via a weighted average, then passed through a logistic
    function scaled so the theoretical maximum yields &asymp;40% probability. Each grid point's
    probability is expanded to a <b>25-mile radius</b> by taking the neighbourhood maximum,
    reflecting the typical uncertainty in mesoscale convective positioning.
  </p>
</div>
""")

parts.append("""<div class="archive-section">
  <div class="section-title">Map Archive</div>
  <table>
    <thead>
      <tr>
        <th>HRRR Run Time (UTC)</th>
        <th>GeoJSON by Forecast Hour</th>
        <th>Interactive Map</th>
      </tr>
    </thead>
    <tbody>
""" + rows_html + """
    </tbody>
  </table>
</div>
""")

parts.append("""<footer>
  Data: <a href="https://rapidrefresh.noaa.gov/hrrr/" style="color:var(--accent)">NOAA HRRR</a>
  &nbsp;|&nbsp;
  Visualisation: <a href="https://python-visualization.github.io/folium/" style="color:var(--accent)">Folium</a>
  &nbsp;|&nbsp;
  Computed via <a href="https://herbie.readthedocs.io/" style="color:var(--accent)">Herbie</a>
  &nbsp;|&nbsp;
  Last rebuilt: """ + generated_at + """
  &nbsp;|&nbsp;
  <em>For educational/research use only &ndash; not for operational forecasting.</em>
</footer>
</body>
</html>
""")

INDEX_PATH.write_text("".join(parts))
print("index.html updated ({}) \u2013 {} map(s) listed.".format(generated_at, len(map_files)))
