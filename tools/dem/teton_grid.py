"""
Teton Dam (Idaho) failure of 5 June 1976: reservoir and terrain for the viewer demo.

  node tools/dem/fetch-3dep.mjs teton-40m -111.52 43.90 1200 40
  python tools/dem/teton_grid.py

Data and assumptions:
- Terrain: USGS 3DEP (NAVD88, m), 40 m cells, as fetched; today's terrain, so the canyon is empty
  and the breached embankment stands as it does now.
- Reservoir level at failure 5,301.7 ft; crest 5,332.0 ft; full pool 5,320.0 ft holding 288,250
  acre-ft (Bureau of Reclamation, Teton Basin Project page). Elevations of that era are taken as
  NGVD29 and converted to NAVD88 with NGS VERTCON 3.0 at the dam: +1.100 m (sigma 0.003 m).
- Dam: 43.90972 N, 111.53917 W. The dam axis is the shortest straight crossing of the canyon
  through that point (searched over directions) that seals the water below the level from the
  plain; water upstream of the axis is held at t = 0 and released instantly.
- Check against the record: the stored volume at the full-pool level against 288,250 acre-ft
  (USBR) and at the failure level against the ~240,000 acre-ft released (Scott, 1977, USGS
  OFR 77-507, citing Ray and Kjelstrom, 1977).
Output: examples/viewer/data/teton-bed.f32 (rows south to north), teton-reservoir.u8, teton.json.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
FT = 0.3048
VERTCON = 1.100                                  # NGVD29 -> NAVD88 at the dam (m)
LEVEL = 5301.7 * FT + VERTCON                    # reservoir at failure, NAVD88 (m)
FULL = 5320.0 * FT + VERTCON                     # full pool, NAVD88 (m)
ACRE_FT = 1233.48184                             # m³
DAM = (-111.53917, 43.90972)

meta = json.loads((ROOT / "data/dem/teton-40m.json").read_text(encoding="utf-8"))
N, dx = meta["width"], meta["dx"]
z = np.fromfile(ROOT / "data/dem/teton-40m.f32", np.float32).reshape(N, N)   # rows north to south
x0, y0, x1, y1 = meta["bbox"]
to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
dxa, dya = to_albers.transform(*DAM)
dc, dr = (dxa - x0) / dx - 0.5, (y1 - dya) / dx - 0.5      # dam position in (column, row), fractional
print(f"dam at column {dc:.1f}, row {dr:.1f}; bed there {z[int(round(dr)), int(round(dc))]:.1f} m; level {LEVEL:.2f} m NAVD88")

rows, cols = np.mgrid[0:N, 0:N]

def axis_cells(theta, half_length=900.0):
    """Cells within one cell of the straight segment through the dam at angle theta (radians)."""
    ux, uy = np.cos(theta), np.sin(theta)
    px, py = cols - dc, rows - dr
    along = px * ux + py * uy
    across = np.abs(-px * uy + py * ux)
    return (np.abs(along) <= half_length / dx) & (across <= 0.75)

def fill(level, wall, seed):
    """4-connected cells below `level` reachable from `seed` without crossing `wall`."""
    inside = np.zeros_like(wall)
    if z[seed] >= level or wall[seed]:
        return inside, False
    queue, inside[seed] = deque([seed]), True
    leaked = False
    while queue:
        r, c = queue.popleft()
        if r in (0, N - 1) or c in (0, N - 1):
            leaked = True
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < N and 0 <= cc < N and not inside[rr, cc] and not wall[rr, cc] and z[rr, cc] < level:
                inside[rr, cc] = True
                queue.append((rr, cc))
    return inside, leaked

# Upstream seed: the lowest cell 1 to 3 km from the dam on the canyon side (east of the dam).
dist = np.hypot(cols - dc, rows - dr) * dx
ring = (dist > 1000) & (dist < 3000) & (cols > dc)
seed = np.unravel_index(np.argmin(np.where(ring, z, np.inf)), z.shape)
print(f"upstream seed at row {seed[0]}, column {seed[1]}, bed {z[seed]:.1f} m")

best = None
for deg in range(0, 180, 3):
    wall = axis_cells(np.radians(deg))
    reservoir, leaked = fill(LEVEL, wall, seed)
    if leaked or reservoir.sum() < 1000:
        continue
    crossing = int((wall & (z < LEVEL)).sum())         # axis cells below the level: the canyon width
    if best is None or crossing < best[1]:
        best = (deg, crossing, wall, reservoir)
if best is None:
    raise SystemExit("no straight axis through the dam seals the reservoir; check the dam coordinates")
deg, crossing, wall, reservoir = best
volume = float(((LEVEL - z) * reservoir).sum() * dx * dx)
full, full_leak = fill(FULL, wall, seed)
full_volume = float(((FULL - z) * full).sum() * dx * dx) if not full_leak else float("nan")
print(f"axis {deg} degrees from east, {crossing} cells ({crossing * dx:.0f} m) below the level")
print(f"reservoir at failure: {reservoir.sum()} cells, {volume / 1e6:.1f} million m³ = {volume / ACRE_FT:,.0f} acre-ft "
      f"(released per USGS: ~240,000 acre-ft)")
print(f"full pool ({FULL:.2f} m): {full_volume / 1e6:.1f} million m³ = {full_volume / ACRE_FT:,.0f} acre-ft (USBR capacity: 288,250 acre-ft)")
rr, cc = np.nonzero(reservoir)
print(f"reservoir extent: rows {rr.min()}..{rr.max()}, columns {cc.min()}..{cc.max()} "
      f"({(cc.max() - cc.min()) * dx / 1000:.1f} km x {(rr.max() - rr.min()) * dx / 1000:.1f} km)")

# Model window: the whole reservoir (its volume drives the flood), the dam, the canyon mouth and the
# first 10 km of the plain beyond the dam, 6 km either side of it; the flood leaves through the edges.
r0 = max(0, min(int(rr.min()) - 20, int(dr) - 150)); r1 = min(N, max(int(rr.max()) + 20, int(dr) + 150))
c0 = max(0, int(dc - 10000 / dx)); c1 = min(N, int(cc.max() + 40))
print(f"window rows {r0}..{r1}, columns {c0}..{c1}: {c1 - c0} x {r1 - r0} cells ({(c1 - c0) * dx / 1000:.1f} km x {(r1 - r0) * dx / 1000:.1f} km)")

out = ROOT / "examples/viewer/data"
bed = z[r0:r1, c0:c1][::-1].astype(np.float32)          # rows south to north
mask = reservoir[r0:r1, c0:c1][::-1].astype(np.uint8)
bed.tofile(out / "teton-bed.f32"); mask.tofile(out / "teton-reservoir.u8")
(out / "teton.json").write_text(json.dumps({
    "width": int(c1 - c0), "height": int(r1 - r0), "dx": dx, "rowOrder": "south-to-north",
    "crs": "EPSG:5070", "units": "m (NAVD88)", "levelAtFailure_NAVD88_m": LEVEL, "fullPool_NAVD88_m": FULL,
    "reservoirVolume_m3": volume, "fullPoolVolume_m3": full_volume, "damAxisDegreesFromEast": deg,
    "sources": {
        "terrain": f"USGS 3DEP, {meta['url'][:120]}..., retrieved {meta['retrieved']}",
        "levels": "Bureau of Reclamation, Teton Basin Project (usbr.gov/projects/index.php?id=414): reservoir 5,301.7 ft at failure, full pool 5,320.0 ft (288,250 acre-ft), crest 5,332.0 ft",
        "datum": "NGVD29 assumed for the 1976 elevations; +1.100 m to NAVD88 (NGS VERTCON 3.0 at 43.90972 N, 111.53917 W)",
        "released": "~240,000 acre-ft over about 8 hours (Scott, 1977, USGS OFR 77-507, citing Ray and Kjelstrom, 1977); flood surge 2.3 million ft³/s (Williams, 1979, USGS OFR 79-1335)",
    }}, indent=2), encoding="utf-8")
print("wrote", out / "teton-bed.f32")
