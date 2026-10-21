"""
Grand Isle, Louisiana: topobathymetry and building footprints for the coastal wave demo.

  python tools/dem/fetch_grandisle.py

1. Bed: NOAA NCEI CUDEM 1/9 arc-second topobathymetric tiles (NAVD88, m; public domain; Amante et
   al., 2023), read as a window across the two tiles that meet at 29.25 N (Cloud-Optimized GeoTIFF
   over HTTP), on their native grid (no resampling): 1/9 arc-second, about 3.0 m east-west and
   3.4 m north-south here. Written as data/dem/grandisle-cudem.f32 (rows north to south) and .json.
2. Buildings and breakwaters: OpenStreetMap (ODbL, (c) OpenStreetMap contributors) through the
   Overpass API, saved as data/osm/grandisle.json, and rasterized to solid cells (cell centres
   inside a building outline; breakwater lines widened to one cell either side) in
   data/dem/grandisle-cudem.solid.u8 (same grid and row order as the bed).
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
WEST, EAST, SOUTH, NORTH = -89.992, -89.952, 29.2255, 29.2705  # about 3.9 km x 5.0 km around the east of the island
BASE = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/LA_MS/"
TILES = ["ncei19_n29x50_w090x00_2020v1.tif", "ncei19_n29x25_w090x00_2020v1.tif"]  # north, south of 29.25 N

with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
    with rasterio.open("/vsicurl/" + BASE + TILES[0]) as north:
        res = north.res[0]
        # Snap the window to the native pixel grid of the tiles (they share it).
        col0 = int(round((WEST - north.bounds.left) / res)); col1 = int(round((EAST - north.bounds.left) / res))
        top = north.bounds.top
        row0 = int(round((top - NORTH) / res)); row1 = int(round((top - SOUTH) / res))
    width, height = col1 - col0, row1 - row0
    bed = np.full((height, width), np.nan, np.float32)
    for name in TILES:
        with rasterio.open("/vsicurl/" + BASE + name) as src:
            assert abs(src.res[0] - res) < 1e-12
            offset = int(round((top - src.bounds.top) / res))  # rows of this tile below the window grid's origin
            r0, r1 = max(row0, offset), min(row1, offset + src.height)
            if r1 <= r0:
                continue
            window = rasterio.windows.Window(col0, r0 - offset, width, r1 - r0)
            data = src.read(1, window=window).astype(np.float32)
            data[data == src.nodata] = np.nan
            bed[r0 - row0:r1 - row0] = np.where(np.isnan(bed[r0 - row0:r1 - row0]), data, bed[r0 - row0:r1 - row0])
    if np.isnan(bed).any():
        raise SystemExit(f"{int(np.isnan(bed).sum())} cells without data in the window")

west, north_edge = north.bounds.left + col0 * res, top - row0 * res
lat = north_edge - height * res / 2
dx = float(np.radians(res) * 6378137 * np.cos(np.radians(lat)) / np.sqrt(1 - 0.00669438 * np.sin(np.radians(lat)) ** 2))
dy = float(np.radians(res) * 6378137 * (1 - 0.00669438) / (1 - 0.00669438 * np.sin(np.radians(lat)) ** 2) ** 1.5)

dem = ROOT / "data" / "dem"
bed.tofile(dem / "grandisle-cudem.f32")
meta = {"name": "grandisle-cudem", "source": "NOAA NCEI CUDEM 1/9 arc-second topobathymetric tiles (doi:10.25921/ds9v-ky35)",
        "tiles": [BASE + t for t in TILES], "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "crs": "EPSG:4269 (NAD83 geographic), native 1/9 arc-second grid", "bbox": [west, north_edge - height * res, west + width * res, north_edge],
        "width": width, "height": height, "dx": dx, "dy": dy, "rowOrder": "north-to-south", "units": "m (NAVD88)",
        "min": float(bed.min()), "max": float(bed.max()),
        "note": "dx and dy are the metric lengths of 1/9 arc-second at the window's central latitude (GRS 1980)."}
(dem / "grandisle-cudem.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"bed {width} x {height}, dx {dx:.3f} m, dy {dy:.3f} m, {bed.min():.2f} to {bed.max():.2f} m NAVD88")

query = f"""[out:json][timeout:90];
(way["building"]({SOUTH},{WEST},{NORTH},{EAST}); way["man_made"~"breakwater|groyne"]({SOUTH},{WEST},{NORTH},{EAST}););
out geom;"""
osm = None
for attempt in range(6):  # the public Overpass servers are often busy; try the main one and a mirror
    endpoint = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"][attempt % 2]
    request = urllib.request.Request(endpoint, data=urllib.parse.urlencode({"data": query}).encode(),
                                     headers={"User-Agent": "FluxCell demo data (github.com/uihilab/FluxCell)"})
    try:
        osm = json.loads(urllib.request.urlopen(request, timeout=180).read()); break
    except Exception as error:  # noqa: BLE001
        print(f"{endpoint}: {error}; retrying"); time.sleep(10)
if osm is None:
    raise SystemExit("Overpass unavailable")
(ROOT / "data" / "osm").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "osm" / "grandisle.json").write_text(json.dumps({"attribution": "(c) OpenStreetMap contributors, ODbL 1.0",
    "query": query, "retrieved": meta["retrieved"], "elements": osm["elements"]}), encoding="utf-8")

# Rasterize: cell centres (lon, lat) of the bed grid.
lons = west + (np.arange(width) + 0.5) * res
lats = north_edge - (np.arange(height) + 0.5) * res
solid = np.zeros((height, width), np.uint8)
buildings = walls = 0
for element in osm["elements"]:
    geometry = element.get("geometry")
    if not geometry:
        continue
    xs = np.array([p["lon"] for p in geometry]); ys = np.array([p["lat"] for p in geometry])
    tags = element.get("tags", {})
    if "building" in tags and len(xs) >= 4 and xs[0] == xs[-1] and ys[0] == ys[-1]:
        c = np.flatnonzero((lons >= xs.min()) & (lons <= xs.max())); r = np.flatnonzero((lats >= ys.min()) & (lats <= ys.max()))
        if not c.size or not r.size:
            continue
        X, Y = np.meshgrid(lons[c], lats[r])
        inside = np.zeros(X.shape, bool)
        for k in range(len(xs) - 1):  # even-odd ray casting
            x0, y0, x1, y1 = xs[k], ys[k], xs[k + 1], ys[k + 1]
            cross = ((y0 > Y) != (y1 > Y)) & (X < (x1 - x0) * (Y - y0) / (y1 - y0 + 1e-300) + x0)
            inside ^= cross
        solid[np.ix_(r, c)] |= inside.astype(np.uint8)
        buildings += 1
    elif tags.get("man_made") in ("breakwater", "groyne"):
        for k in range(len(xs) - 1):  # sample the line every third of a cell, one cell either side
            n = int(max(abs(xs[k + 1] - xs[k]), abs(ys[k + 1] - ys[k])) / res * 3) + 1
            for t in np.linspace(0, 1, n):
                ci = int(np.floor((xs[k] + t * (xs[k + 1] - xs[k]) - west) / res)); ri = int(np.floor((north_edge - (ys[k] + t * (ys[k + 1] - ys[k]))) / res))
                if 0 <= ri < height and 0 <= ci < width:  # points outside the window are skipped
                    solid[max(0, ri - 1):ri + 2, max(0, ci - 1):ci + 2] = 1
        walls += 1
solid.tofile(dem / "grandisle-cudem.solid.u8")
print(f"OSM: {buildings} buildings, {walls} breakwaters or groynes; {int(solid.sum())} solid cells")
