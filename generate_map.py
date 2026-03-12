"""
Tornado Probability Interactive Map Generator
=============================================
Generates a multi-layer Folium HTML map for the Northeast US showing
tornado probability (%) at each HRRR forecast hour (+01 to +06).

Uses HeatMap layers for smooth, compact rendering.  Click anywhere on the
map to query the nearest grid-point probability via embedded JSON data.

Usage:
    python generate_map.py [--run-time "YYYY-MM-DD HH:MM"] [--output-dir maps]
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import folium
import numpy as np
from folium.plugins import HeatMap

import tornado_probability as tp


# HeatMap gradient: low → high probability
HEATMAP_GRADIENT = {
    0.00: "#00000000",
    0.08: "#64c864cc",
    0.30: "#ffff00dd",
    0.55: "#ffa500ee",
    0.78: "#dc3232ff",
    1.00: "#640064ff",
}

HEATMAP_STRIDE = 3   # keep 1 in 3 grid points for heatmap rendering
HEATMAP_MIN_PROB = 0.01

LEGEND_HTML = """
<div id="prob-legend" style="
    position:fixed;bottom:30px;right:20px;z-index:9999;
    background:rgba(13,17,23,0.93);color:#c9d1d9;
    border:1.5px solid #30363d;border-radius:8px;
    padding:10px 14px;font-family:Arial,sans-serif;font-size:12px;
    box-shadow:3px 3px 8px rgba(0,0,0,0.5);">
  <b style="font-size:13px;color:#f0f0f0;">Tornado Probability</b><br>
  <i style="font-size:10px;color:#8b949e;">25-mi radius &bull; HRRR-based</i>
  <div style="margin-top:8px;line-height:1.9;">
    <span style="display:inline-block;width:13px;height:13px;background:#64c864;border-radius:2px;vertical-align:middle"></span>&nbsp; 2&ndash;5 %<br>
    <span style="display:inline-block;width:13px;height:13px;background:#ffff00;border-radius:2px;vertical-align:middle"></span>&nbsp; 5&ndash;10 %<br>
    <span style="display:inline-block;width:13px;height:13px;background:#ffa500;border-radius:2px;vertical-align:middle"></span>&nbsp; 10&ndash;15 %<br>
    <span style="display:inline-block;width:13px;height:13px;background:#dc3232;border-radius:2px;vertical-align:middle"></span>&nbsp; 15&ndash;25 %<br>
    <span style="display:inline-block;width:13px;height:13px;background:#640064;border-radius:2px;vertical-align:middle"></span>&nbsp; &gt; 25 %
  </div>
  <div style="margin-top:8px;font-size:10px;color:#8b949e;">Click map for local probability</div>
</div>
"""

INFO_HTML = """
<div id="prob-info" style="
    position:fixed;top:10px;right:20px;z-index:9999;
    background:rgba(13,17,23,0.93);color:#c9d1d9;
    border:1.5px solid #30363d;border-radius:8px;
    padding:10px 14px;font-family:Arial,sans-serif;font-size:11px;
    max-width:240px;box-shadow:3px 3px 8px rgba(0,0,0,0.5);">
  <b style="font-size:13px;color:#f0f0f0;">HRRR Tornado Probability</b><br>
  <i style="color:#8b949e;">Northeast US &bull; 6-Hour Outlook</i><br><br>
  <b>Index components:</b><br>
  &bull; CAPE (20%)<br>
  &bull; 0&ndash;6 km bulk shear (20%)<br>
  &bull; Sig. Tornado Parameter (30%)<br>
  &bull; 0&ndash;3 km UH 1-hr max (30%)<br><br>
  Toggle <b>forecast hours</b> via the layer control (top-left).<br>
  <b>Click</b> anywhere on the map for the local probability estimate.
</div>
"""


def build_geojson(lat, lon, prob, fxx, valid_str, run_str):
    features = []
    for la, lo, pr in zip(lat, lon, prob):
        pct = round(float(pr) * 100, 1)
        if pct < 1.0:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lo), float(la)]},
            "properties": {"fxx": fxx, "run_time": run_str,
                           "valid_time": valid_str, "probability_pct": pct},
        })
    return {"type": "FeatureCollection", "features": features}


def build_map(results, run_time, output_path):
    run_str = run_time.strftime("%Y-%m-%d %H:%M UTC")
    m = folium.Map(
        location=[42.5, -74.5], zoom_start=6,
        tiles=None, prefer_canvas=True,
    )

    folium.TileLayer("CartoDB positron", name="Light basemap", show=True).add_to(m)
    folium.TileLayer("OpenStreetMap",    name="OpenStreetMap",  show=False).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark basemap", show=False).add_to(m)

    # Compact lookup table embedded in HTML for click-to-query
    # Structure: { fxx: { "lats": [...], "lons": [...], "probs": [...] } }
    # Downsampled to every 4th point to keep JSON small
    lookup_stride = 4
    lookup_data = {}
    geojson_links = []

    for res in results:
        fxx = res["fxx"]
        valid_dt = datetime.fromisoformat(res["valid_time"])
        valid_str = valid_dt.strftime("%Y-%m-%d %H:%M UTC")

        lat  = res["lat"]
        lon  = res["lon"]
        prob = res["probability"]

        layer_name = "+{:02d}h  Valid: {}".format(fxx, valid_str)
        fg = folium.FeatureGroup(name=layer_name, show=(fxx == 1))

        # HeatMap
        mask = prob >= HEATMAP_MIN_PROB
        lat_h  = lat[mask][::HEATMAP_STRIDE]
        lon_h  = lon[mask][::HEATMAP_STRIDE]
        prob_h = prob[mask][::HEATMAP_STRIDE]
        weights = np.clip(prob_h / 0.40, 0, 1).tolist()
        heat_data = [[float(a), float(o), float(w)]
                     for a, o, w in zip(lat_h, lon_h, weights)]
        HeatMap(
            heat_data,
            gradient=HEATMAP_GRADIENT,
            radius=20, blur=25,
            min_opacity=0.0, max_zoom=10,
        ).add_to(fg)
        fg.add_to(m)

        # Lookup table (sparser grid, round to 2 dp to save space)
        lk_lat  = [round(float(v), 2) for v in lat[::lookup_stride]]
        lk_lon  = [round(float(v), 2) for v in lon[::lookup_stride]]
        lk_prob = [round(float(v) * 100, 1) for v in prob[::lookup_stride]]
        lookup_data[str(fxx)] = {"lats": lk_lat, "lons": lk_lon, "probs": lk_prob,
                                  "valid": valid_str}

        # GeoJSON export
        gj = build_geojson(lat, lon, prob, fxx, valid_str, run_str)
        gj_name = (output_path.parent /
                   "tornado_prob_{run}_fxx{fxx:02d}.geojson".format(
                       run=run_time.strftime("%Y%m%d_%H%M"), fxx=fxx))
        with open(gj_name, "w") as f:
            json.dump(gj, f, separators=(",", ":"))
        geojson_links.append(gj_name.name)
        print("  GeoJSON \u2192 {}".format(gj_name.name))

    # ------------------------------------------------------------------
    # Click-to-query JavaScript
    # Finds nearest grid point in the active layer's lookup table and
    # displays a popup with the local tornado probability.
    # ------------------------------------------------------------------
    click_js = """
<script>
(function() {
  var LOOKUP = """ + json.dumps(lookup_data, separators=(",", ":")) + """;
  var RUN    = '""" + run_str + """';

  function haversine(la1, lo1, la2, lo2) {
    var R = 6371, r = Math.PI/180;
    var dLat = (la2-la1)*r, dLon = (lo2-lo1)*r;
    var a = Math.sin(dLat/2)*Math.sin(dLat/2) +
            Math.cos(la1*r)*Math.cos(la2*r)*Math.sin(dLon/2)*Math.sin(dLon/2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  }

  function activeFxx() {
    // Try to detect which FeatureGroup layer is currently shown.
    // Layer names have pattern "+NNh  Valid: ..."
    var best = '1';
    document.querySelectorAll('.leaflet-control-layers-selector').forEach(function(el) {
      if (el.checked) {
        var label = el.parentNode.textContent.trim();
        var m = label.match(/\\+(\\d+)h/);
        if (m) best = m[1];
      }
    });
    return best;
  }

  document.addEventListener('DOMContentLoaded', function() {
    var mapEl = document.querySelector('#map') ||
                document.querySelector('.folium-map') ||
                document.querySelector('div[id^="map_"]');
    if (!mapEl) return;
    var leafletMap = null;
    // Grab the Leaflet map object from the global window variable Folium creates
    for (var k in window) {
      if (window[k] && window[k]._leaflet_id !== undefined && window[k].on) {
        leafletMap = window[k]; break;
      }
    }
    if (!leafletMap) return;

    leafletMap.on('click', function(e) {
      var clickLat = e.latlng.lat, clickLon = e.latlng.lng;
      var fxx = activeFxx();
      var data = LOOKUP[fxx];
      if (!data) return;

      // Find nearest grid point (brute-force; lookup is ~5k pts, fast enough)
      var bestIdx = 0, bestDist = Infinity;
      for (var i=0; i<data.lats.length; i++) {
        var d = haversine(clickLat, clickLon, data.lats[i], data.lons[i]);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }

      var prob = data.probs[bestIdx];
      var color = prob < 2  ? '#64c864'
                : prob < 5  ? '#b8e86e'
                : prob < 10 ? '#ffff00'
                : prob < 15 ? '#ffa500'
                : prob < 25 ? '#dc3232'
                :             '#640064';

      var content =
        '<div style="font-family:Arial,sans-serif;font-size:12px;min-width:180px;">' +
        '<b style="font-size:14px;">Tornado Probability</b><br>' +
        'Run: ' + RUN + '<br>' +
        'Valid: +' + fxx.padStart(2,'0') + 'h &rarr; ' + data.valid + '<br>' +
        'Nearest grid point: ' + data.lats[bestIdx] + '&deg;N, ' + data.lons[bestIdx] + '&deg;<br>' +
        'Distance: ' + Math.round(bestDist) + ' km<br>' +
        '<span style="font-size:18px;font-weight:bold;color:' + color + ';">' + prob.toFixed(1) + '%</span>' +
        ' tornado probability' +
        '</div>';

      L.popup().setLatLng(e.latlng).setContent(content).openOn(leafletMap);
    });
  });
})();
</script>
"""

    # Panels & click handler
    title_html = (
        "<div style='position:fixed;top:0;left:0;right:0;z-index:10000;"
        "background:rgba(13,17,30,0.88);color:white;text-align:center;"
        "padding:6px 0;font-family:Arial,sans-serif;font-size:14px;"
        "font-weight:bold;letter-spacing:0.3px;'>"
        "HRRR Tornado Probability &bull; Northeast US &bull; Run: {run} &bull; "
        "6-Hour Outlook</div>"
    ).format(run=run_str)

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    m.get_root().html.add_child(folium.Element(INFO_HTML))
    m.get_root().html.add_child(folium.Element(click_js))

    folium.LayerControl(collapsed=False, position="topleft").add_to(m)

    m.save(str(output_path))
    sz = output_path.stat().st_size / 1e6
    print("\nMap saved \u2192 {}  ({:.1f} MB)".format(output_path.name, sz))
    return geojson_links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-time",
                        default=datetime.utcnow().strftime("%Y-%m-%d %H:00"))
    parser.add_argument("--output-dir", default="maps")
    args = parser.parse_args()

    run_time = datetime.strptime(args.run_time, "%Y-%m-%d %H:%M")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== HRRR Tornado Probability Map Generator ===")
    print("Run time  : {}".format(run_time.strftime("%Y-%m-%d %H:%M UTC")))
    print("Output dir: {}".format(output_dir.resolve()))

    results = tp.run(run_time, output_dir)

    map_filename = "tornado_prob_{}.html".format(run_time.strftime("%Y%m%d_%H%M"))
    build_map(results, run_time, output_dir / map_filename)


if __name__ == "__main__":
    main()
    print("\nDone.")
