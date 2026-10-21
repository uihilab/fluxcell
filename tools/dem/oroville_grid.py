"""
Oroville Dam spillways, 12 February 2017: terrain, lake and release cells for the viewer demo.

  node tools/dem/fetch-3dep.mjs oroville-2m -121.494 39.540 1024 2
  python tools/dem/oroville_grid.py

Data and assumptions:
- Terrain: USGS 3DEP (NAVD88, m), 2 m cells; today's terrain, with the rebuilt main spillway and
  the emergency spillway's new buttress. The lidar sees the lake at about 205 m; the lake below that
  surface is not represented (it does not take part in the spill).
- Lake level: CDEC station ORO (DWR), hourly reservoir elevation, peak 902.59 ft at 03:00 on
  12 February 2017, above the 901 ft emergency spillway crest from 08:00 on 11 February to 20:00
  on 12 February. Project elevations are NGVD29: VERTCON 3.0 at the spillway gives +0.706 m, so
  901 ft is 275.33 m NAVD88 and the measured crest of the weir in the lidar has a median of 275.36 m
  (checked below); the peak is 275.81 m NAVD88.
- Main spillway: the gates (the thin headworks structure in the lidar, whose pier gaps would let the
  lake through) are made solid, and the release enters as a source over the chute cells just
  downstream of them: 55,000 ft³/s, the CDEC reservoir outflow at the time of the peak.
- Lake: the cells reachable from the lake below 274.9 m (the lowest point of the spill path is
  274.94 m) with the gates closed, filled to the peak level and held there at the edges they touch.
Output: examples/viewer/data/oroville-bed.f32 (rows south to north), oroville-mask.u8 (1 lake,
2 release, 3 solid), oroville.json.
"""
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FT, VERTCON = 0.3048, 0.706
CREST, PEAK = 901 * FT + VERTCON, 902.59 * FT + VERTCON
RELEASE = 55000 * FT ** 3                                       # m³/s
meta = json.loads((ROOT / "data/dem/oroville-2m.json").read_text(encoding="utf-8"))
N, dx = meta["width"], meta["dx"]
z = np.fromfile(ROOT / "data/dem/oroville-2m.f32", np.float32).reshape(N, N)   # rows north to south
rows, cols = np.mgrid[0:N, 0:N]

def band(a, b, near, far):
    """Cells whose distance from segment a-b lies in [near, far] cells on the side of +normal."""
    (r0, c0), (r1, c1) = a, b
    d = np.array([r1 - r0, c1 - c0], float); L = np.hypot(*d); d /= L
    pr, pc = rows - r0, cols - c0
    t = pr * d[0] + pc * d[1]; n = pr * d[1] - pc * d[0]          # n > 0: down-left of the gate line
    return (t >= -1) & (t <= L + 1) & (n >= near) & (n <= far)

GATE = ((333, 611), (408, 652))
gates = band(*GATE, -1.6, 1.6)
release = band(*GATE, 12, 20) & (z < 255)                         # chute floor below the gate structure

def fill(level, seed):
    inside = np.zeros_like(gates); q = deque([seed]); inside[seed] = True
    while q:
        r, c = q.popleft()
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < N and 0 <= cc < N and not inside[rr, cc] and not gates[rr, cc] and z[rr, cc] < level:
                inside[rr, cc] = True; q.append((rr, cc))
    return inside

LAKE_SEED = (150, 830)
lake = fill(274.9, LAKE_SEED)
# Spill elevation: the lowest maximum elevation along any path from the lake to the river.
best = np.full(z.shape, np.inf); best[LAKE_SEED] = z[LAKE_SEED]; heap = [(float(z[LAKE_SEED]), LAKE_SEED)]
while heap:
    e, (r, c) = heapq.heappop(heap)
    if e > best[r, c]: continue
    if (r, c) == (1000, 100): break
    for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
        if 0 <= rr < N and 0 <= cc < N and not gates[rr, cc] and max(e, z[rr, cc]) < best[rr, cc]:
            best[rr, cc] = max(e, z[rr, cc]); heapq.heappush(heap, (best[rr, cc], (rr, cc)))
spill = float(best[1000, 100])
# Weir crest: along the straight emergency spillway weir north of the headworks, the highest cell
# across the wall at each of 40 stations.
crest = [float(z[int(round(r)), int(round(c)) - 4:int(round(c)) + 5].max()) for r, c in zip(np.linspace(110, 320, 40), np.linspace(557, 608, 40))]
print(f"spill elevation {spill:.2f} m; weir crest median {np.median(crest):.2f} m (901 ft NGVD29 = {CREST:.2f} m NAVD88); peak level {PEAK:.2f} m")
print(f"lake {int(lake.sum())} cells, release {int(release.sum())} cells, gates {int(gates.sum())} cells")
if lake[600, 250] or lake[1000, 100]:
    raise SystemExit("the lake leaks past the gates; check the gate line")

# Window: the lake side to the east, the spillways and the river below (rows 0..899, columns 150..).
r0, r1, c0, c1 = 0, 900, 150, N
mask = np.zeros((N, N), np.uint8); mask[lake] = 1; mask[release] = 2; mask[gates] = 3
out = ROOT / "examples/viewer/data"
z[r0:r1, c0:c1][::-1].astype(np.float32).tofile(out / "oroville-bed.f32")
mask[r0:r1, c0:c1][::-1].tofile(out / "oroville-mask.u8")
(out / "oroville.json").write_text(json.dumps({
    "width": c1 - c0, "height": r1 - r0, "dx": dx, "rowOrder": "south-to-north", "units": "m (NAVD88)",
    "peak_NAVD88_m": PEAK, "crest_NAVD88_m": CREST, "spillElevation_m": spill, "weirCrestMedian_m": float(np.median(crest)),
    "release_m3s": RELEASE,
    "sources": {
        "terrain": f"USGS 3DEP, retrieved {meta['retrieved']}",
        "levels": "CDEC station ORO (California DWR), hourly reservoir elevation (sensor 6), peak 902.59 ft at 2017-02-12 03:00; outflow (sensor 23) 55,070 ft³/s at that time",
        "datum": "project elevations NGVD29; +0.706 m to NAVD88 (NGS VERTCON 3.0 at 39.545 N, 121.49 W)",
        "crest": "emergency spillway weir crest 901 ft (California Water Commission, 2023 Oroville handout)",
    }}, indent=2), encoding="utf-8")
print("wrote", out / "oroville-bed.f32", f"{c1 - c0} x {r1 - r0}")
