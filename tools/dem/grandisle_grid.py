"""
Coast-aligned 2 m grid for the Grand Isle demo, from the fetched CUDEM window and OSM data
(tools/dem/fetch_grandisle.py):

  python tools/dem/grandisle_grid.py

The island and its Gulf shoreline run south-west to north-east, so waves from the Gulf would
cross the edges of a north-aligned grid obliquely. The grid is rotated so that its x axis follows
the shoreline (the principal direction of the land cells) and its south edge lies in the Gulf; an
incident-wave boundary on that edge then generates plane waves that reach the shore head on. The
window runs from 560 m Gulfward of the dune crest (about 400 m offshore of the beach; the nearshore
breakwaters lie 100 to 300 m out) to 540 m inland of it. The bed is resampled bilinearly from the native 1/9 arc-second grid.
Buildings (OSM outlines) and breakwaters (OSM lines widened to 9 m) are described by their signed
distance (negative inside); cells whose centre lies inside are solid, and the viewer draws the
outlines from the distance field, so they keep their shape between cell centres.
Output: examples/viewer/data/grandisle-bed.f32 (rows south to north), grandisle-solid.u8,
grandisle-shape.i8 (signed distance in 1/16 cell, clamped to 8 cells), grandisle.json.
"""
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

ROOT = Path(__file__).resolve().parents[2]
DX, ALONG, V0, V1 = 2.0, 1400.0, -560.0, 540.0  # m: cell, along-shore length, cross-shore range from the dune crest (v < 0: Gulf side)
BREAKWATER_HALF_WIDTH = 4.5  # m

meta = json.loads((ROOT / "data/dem/grandisle-cudem.json").read_text(encoding="utf-8"))
W, H = meta["width"], meta["height"]
bed = np.fromfile(ROOT / "data/dem/grandisle-cudem.f32", np.float32).reshape(H, W)
west, south, east, north = meta["bbox"]
res = (east - west) / W
lat0 = (south + north) / 2
mx, my = meta["dx"] / res, meta["dy"] / res        # metres per degree of longitude and latitude here
to_xy = lambda lon, lat: ((np.asarray(lon) - west) * mx, (north - np.asarray(lat)) * -my)  # x east, y north (y <= 0)

# Native cell centres in metres (x east from the west edge, y north from the north edge, so y <= 0).
xc = (np.arange(W) + 0.5) * meta["dx"]; yc = -(np.arange(H) + 0.5) * meta["dy"]
X, Y = np.meshgrid(xc, yc)
land = (bed > 0.1) & (bed < 4)
px, py = X[land], Y[land]
centre = np.array([px.mean(), py.mean()])
cov = np.cov(np.vstack([px - centre[0], py - centre[1]]))
along = np.linalg.eigh(cov)[1][:, -1]                 # principal direction of the land
cross = np.array([-along[1], along[0]])
deep = bed < -3
if np.mean((X[deep] - centre[0]) * cross[0] + (Y[deep] - centre[1]) * cross[1]) > 0:
    cross = -cross                                    # cross-shore axis points from the Gulf to the bay
if along[0] < 0:
    along = -along
angle = float(np.degrees(np.arctan2(along[1], along[0])))
print(f"shoreline direction {angle:.1f} degrees from east; centre {centre.round(1)} m")

def sample(along, cross, u, v):
    """Bilinear sample of the native grid at the rotated-grid points (u along, v across)."""
    U, V = np.meshgrid(u, v)
    gx = centre[0] + U * along[0] + V * cross[0]; gy = centre[1] + U * along[1] + V * cross[1]
    fi = gx / meta["dx"] - 0.5; fj = -gy / meta["dy"] - 0.5
    i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int)
    if i0.min() < 0 or j0.min() < 0 or i0.max() + 1 >= W or j0.max() + 1 >= H:
        raise SystemExit("the rotated window leaves the fetched data; widen fetch_grandisle.py")
    a = fi - i0; b = fj - j0
    return ((1 - a) * (1 - b) * bed[j0, i0] + a * (1 - b) * bed[j0, i0 + 1] + (1 - a) * b * bed[j0 + 1, i0] + a * b * bed[j0 + 1, i0 + 1]).astype(np.float32)

# The island axis is not quite the axis of this stretch of beach: fit the dune crest (the highest
# point of each column between 900 and 300 m Gulfward of the centre) and turn the grid onto it.
nx = int(ALONG / DX)
u = (np.arange(nx) + 0.5) * DX - ALONG / 2
probe = np.arange(-900.0, -300.0, DX)
crest = probe[np.argmax(sample(along, cross, u, probe), axis=0)]
slope_, crest0 = np.polyfit(u, crest, 1)
theta = np.arctan(slope_)
along, cross = np.cos(theta) * along + np.sin(theta) * cross, -np.sin(theta) * along + np.cos(theta) * cross
centre = centre + crest0 * cross / np.cos(theta)   # v = 0 now lies on the fitted crest
angle = float(np.degrees(np.arctan2(along[1], along[0])))
print(f"beach direction {angle:.1f} degrees from east (turned {np.degrees(theta):.1f}); crest {crest0:.0f} m from the island centre")
ny = int((V1 - V0) / DX)
v = (np.arange(ny) + 0.5) * DX + V0                  # cross-shore distance from the dune crest; row 0 is the Gulf edge
z = sample(along, cross, u, v)

osm = json.loads((ROOT / "data/osm/grandisle.json").read_text(encoding="utf-8"))
def to_uv(geometry):
    x, y = to_xy([p["lon"] for p in geometry], [p["lat"] for p in geometry])
    dx, dy = x - centre[0], y - centre[1]
    return dx * along[0] + dy * along[1], dx * cross[0] + dy * cross[1]
# Signed distance (m) to the buildings and breakwaters at the cell centres, computed per shape
# within 8 cells of it and combined by the minimum.
LIMIT = 8 * DX
sdf = np.full((ny, nx), LIMIT, np.float64)
buildings = walls = 0
for element in osm["elements"]:
    g, tags = element.get("geometry"), element.get("tags", {})
    if not g:
        continue
    us, vs = to_uv(g)
    if "building" in tags and len(us) >= 4 and g[0] == g[-1]:
        shape = Polygon(np.column_stack([us, vs])).buffer(0); buildings_hit = True
    elif tags.get("man_made") in ("breakwater", "groyne"):
        shape = LineString(np.column_stack([us, vs])).buffer(BREAKWATER_HALF_WIDTH, cap_style="flat"); buildings_hit = False
    else:
        continue
    x0, y0, x1, y1 = shape.bounds
    c = np.flatnonzero((u >= x0 - LIMIT) & (u <= x1 + LIMIT)); r = np.flatnonzero((v >= y0 - LIMIT) & (v <= y1 + LIMIT))
    if not c.size or not r.size:
        continue
    PU, PV = np.meshgrid(u[c], v[r])
    d = shapely.distance(shape.boundary, shapely.points(PU, PV))
    d = np.where(shapely.contains_xy(shape, PU, PV), -d, d)
    sdf[np.ix_(r, c)] = np.minimum(sdf[np.ix_(r, c)], d)
    if buildings_hit:
        buildings += 1
    else:
        walls += 1
solid = (sdf < 0).astype(np.uint8)
shape_i8 = np.clip(np.round(sdf / DX * 16), -127, 127).astype(np.int8)
out = ROOT / "examples/viewer/data"
z.tofile(out / "grandisle-bed.f32"); solid.tofile(out / "grandisle-solid.u8"); shape_i8.tofile(out / "grandisle-shape.i8")
lon_c = west + centre[0] / mx; lat_c = north + centre[1] / my
(out / "grandisle.json").write_text(json.dumps({
    "width": nx, "height": ny, "dx": DX, "dy": DX, "rowOrder": "south-to-north (row 0 on the Gulf edge)",
    "centre": {"lon": lon_c, "lat": lat_c}, "centreIs": "on the fitted dune crest, at mid-length", "shorelineAngleFromEast_deg": angle,
    "units": "m (NAVD88)", "meanSeaLevel_NAVD88_m": 0.072, "idaPeak_NAVD88_m": 0.072 + 1.566,
    "sources": {
        "bed": "NOAA NCEI CUDEM 1/9 arc-second topobathymetry (doi:10.25921/ds9v-ky35), tiles " + ", ".join(t.split("/")[-1] for t in meta["tiles"]) + "; public domain; resampled bilinearly onto this grid",
        "buildings": f"OpenStreetMap ({buildings} buildings, {walls} breakwaters or groynes widened to {2 * BREAKWATER_HALF_WIDTH:g} m; (c) OpenStreetMap contributors, ODbL), retrieved {osm['retrieved']}",
        "shape": "grandisle-shape.i8: signed distance to the buildings and breakwaters at the cell centres, in 1/16 cell, clamped to 8 cells (negative inside)",
        "meanSeaLevel": "NOAA VDatum: local mean sea level = +0.072 m NAVD88 at 29.24 N, 89.97 W (uncertainty 0.157 m)",
        "idaPeak": "NOAA station 8761724 (Grand Isle): maximum 3.665 m above station datum on 2021-08-29 22:54 GMT, MSL 2.099 m (epoch 2012-2016): 1.566 m above MSL",
    }}, indent=2), encoding="utf-8")
print(f"grid {nx} x {ny} at {DX} m; bed {z.min():.2f} to {z.max():.2f} m; {buildings} buildings, {walls} breakwaters; {int(solid.sum())} solid cells")
for r in range(ny - 1, -1, -20):
    print(f"{v[r]:7.0f} " + "".join("B" if solid[r, c] else ("#" if z[r, c] > 2 else ("+" if z[r, c] > 0.07 else ("~" if z[r, c] > -1.5 else "."))) for c in range(0, nx, 8)))
