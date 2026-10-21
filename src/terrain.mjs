// Real-terrain fixtures on small USGS 3DEP tiles stored in data/dem (see tools/dem/fetch-3dep.mjs).
// Grids are row-major with row 0 at the SOUTH edge (the served rows are flipped on load).

// floodwall: the 17th Street Canal I-wall (crest about +4 m) is too thin for bare-earth lidar; it is
// restored as solid cells along the crest of the canal's east embankment (see canalFloodwall).
// The 2 m tiles come from lidar surveys (survey-locked mosaics; see their .json metadata). crop =
// [x0, y0, width, height] in cells of the stored tile, row 0 at the south edge, applied on load
// (loadDem(name, { crop }) overrides it). The New Orleans breach, in stored-tile cells, marks the
// 2005 17th Street Canal floodwall breach (30.0172 N, 90.1208 W): a 140 m gap on the protected
// (east) side of the canal embankment, with an illustrative constant inflow.
export const DEMS = Object.freeze({
    'yosemite-elcapitan': { label: 'Yosemite Valley / El Capitan (steep scarp)', manning: 0.035,
        lakeLevel: 1215, release: { depth: 5, radius: 250 } },
    'neworleans-lakeview': { label: 'New Orleans / Lakeview (flat, below sea level)', manning: 0.03,
        lakeLevel: -1, inflow: { discharge: 200, cells: 5 } },
    'yosemite-elcapitan-2m': { label: 'Yosemite Valley / El Capitan, lidar 2 m', manning: 0.035,
        lakeLevel: 1215, release: { depth: 5, radius: 150, inset: true } },
    'neworleans-17thstreet-2m': { label: 'New Orleans / 17th Street Canal and Lakeview, lidar 2 m', manning: 0.03,
        lakeLevel: -1, crop: [640, 700, 1024, 1024], breach: { x0: 686, x1: 693, y0: 1591, y1: 1661, discharge: 600 }, floodwall: true },
});

export async function loadDem(name, { crop } = {}) {
    if (!DEMS[name]) throw new RangeError(`Unknown DEM ${name}`);
    // The full tile or, when it is absent (as in a published copy of the viewer), the versioned
    // demo window <name>.demo.* written by tools/nthmp/demo-data.mjs. Crops and the returned origin
    // are in full-tile cells either way.
    const read = async suffix => {
        const base = new URL(`../data/dem/${name}${suffix}`, import.meta.url);
        if (typeof process !== 'undefined' && process.versions?.node && typeof window === 'undefined') {
            const { readFile } = await import('node:fs/promises');
            const file = await readFile(new URL(`${base}.f32`));
            return { meta: JSON.parse(await readFile(new URL(`${base}.json`), 'utf8')), bytes: file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) };
        }
        const [m, b] = await Promise.all([fetch(`${base}.json`), fetch(`${base}.f32`)]);
        if (!m.ok || !b.ok) throw new Error(`${name}${suffix}: HTTP ${m.status}/${b.status}`);
        return { meta: await m.json(), bytes: await b.arrayBuffer() };
    };
    let tile, offset = [0, 0];
    try { tile = await read(''); } catch (error) {
        tile = await read('.demo').catch(() => { throw error; });
        offset = tile.meta.window.origin;
    }
    const { meta, bytes } = tile, { width: W, height: H } = meta, served = new Float32Array(bytes);
    if (served.length !== W * H) throw new Error(`${name}: expected ${W * H} values`);
    const [cx, cy, w, h] = crop ?? DEMS[name].crop ?? [offset[0], offset[1], W, H];
    const x0 = cx - offset[0], y0 = cy - offset[1];
    if (x0 < 0 || y0 < 0 || x0 + w > W || y0 + h > H) throw new RangeError(`${name}: crop outside the available ${W} x ${H} cells`);
    const bed = new Float32Array(w * h);
    for (let y = 0; y < h; y++) bed.set(served.subarray((H - 1 - (y + y0)) * W + x0, (H - 1 - (y + y0)) * W + x0 + w), y * w);
    return { name, meta, bed, width: w, height: h, dx: meta.dx, dy: meta.dy, ...DEMS[name], origin: [cx, cy] };
}

// Solid cells along the east embankment of the 17th Street Canal: in each row the canal is the
// hydro-flattened water surface (the first run of at least five cells within 8 cm of -0.2 m, x < 120) and
// the wall sits at the highest cell within six cells east of it (the crest is 2 to 4 cells from the
// water edge along the canal; a wider window reaches other features near the rebuilt breach). Rows are joined so that the wall
// is watertight across faces.
export function canalFloodwall(dem) {
    const { width: w, height: h, bed } = dem, crest = new Int32Array(h), solid = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
        let end = -1, run = 0;
        // The canal is the first such run from the west (ground east of the embankment can also lie
        // near -0.2 m).
        for (let x = 0; x < 120; x++) {
            if (Math.abs(bed[y * w + x] + 0.2) < 0.08) { if (++run >= 5) end = x; } else if (end >= 0) break; else run = 0;
        }
        if (end < 0) throw new Error(`${dem.name}: no canal found in row ${y}`);
        let best = end;
        for (let x = end; x < end + 6; x++) if (bed[y * w + x] > bed[y * w + best]) best = x;
        crest[y] = best;
    }
    for (let y = 0; y < h; y++) {
        const a = y > 0 ? Math.min(crest[y], crest[y - 1]) : crest[y], b = y > 0 ? Math.max(crest[y], crest[y - 1]) : crest[y];
        for (let x = a; x <= b; x++) solid[y * w + x] = 1;
    }
    return solid;
}

// Kinds: 'lake' (still water at an absolute level, walls), 'rain' (100 mm/h on a dry tile,
// outflow boundaries), 'release' (a column of water on the highest ground, outflow boundaries),
// 'inflow' (a distributed point discharge, closed boundaries), 'breach' (a discharge through the
// tile's breach cells, outflow boundaries). duration is simulated seconds. With tracer: true the
// released column and the inflowing water carry concentration 1 (rain carries none).
export function terrainScenario(dem, kind, { datum = 'auto', tracer = false } = {}) {
    const { width: w, height: h, dx, dy, bed } = dem, n = w * h;
    const state = new Float32Array(4 * n), manning = new Float32Array(n), source = new Float32Array(n);
    // Free-outfall edges: runoff may leave a cut tile but never enters through it.
    const config = { width: w, height: h, dx, dy, gravity: 9.81, cfl: 0.4, datum, boundaryX: 'outflow', boundaryY: 'outflow', tracer };
    let duration;
    if (kind === 'lake') {
        Object.assign(config, { boundaryX: 'wall', boundaryY: 'wall' });
        for (let i = 0; i < n; i++) state[4 * i] = Math.max(0, dem.lakeLevel - bed[i]);
        duration = 600;
    } else if (kind === 'rain') {
        manning.fill(dem.manning);
        config.rain = 100e-3 / 3600;
        duration = 900;
    } else if (kind === 'release') {
        if (!dem.release) throw new RangeError(`${dem.name} has no release case`);
        manning.fill(dem.manning);
        // Highest cell; with release.inset, the highest whose column lies wholly inside the tile.
        const r = dem.release.radius, mx = dem.release.inset ? Math.ceil(r / dx) + 2 : 0, my = dem.release.inset ? Math.ceil(r / dy) + 2 : 0;
        let top = my * w + mx;
        for (let y = my; y < h - my; y++) for (let x = mx; x < w - mx; x++) if (bed[y * w + x] > bed[top]) top = y * w + x;
        const tx = top % w, ty = Math.floor(top / w);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            if (((x - tx) * dx) ** 2 + ((y - ty) * dy) ** 2 <= r * r) state[4 * (y * w + x)] = dem.release.depth;
        }
        duration = 300;
    } else if (kind === 'inflow') {
        if (!dem.inflow) throw new RangeError(`${dem.name} has no inflow case`);
        Object.assign(config, { boundaryX: 'wall', boundaryY: 'wall' });
        manning.fill(dem.manning);
        // Breach-like inflow on the lowest cells of the northern edge band.
        const k = dem.inflow.cells, candidates = [];
        for (let x = k; x < w - k; x++) candidates.push([bed[(h - 1 - k) * w + x], x]);
        candidates.sort((a, b) => a[0] - b[0]);
        const cx = candidates[0][1], cy = h - 1 - k, rate = dem.inflow.discharge / (k * k * dx * dy);
        for (let y = cy - (k >> 1); y <= cy + (k >> 1); y++) for (let x = cx - (k >> 1); x <= cx + (k >> 1); x++) source[y * w + x] = rate;
        duration = 1800;
    } else if (kind === 'breach') {
        if (!dem.breach) throw new RangeError(`${dem.name} has no breach case`);
        manning.fill(dem.manning);
        const [ox, oy] = dem.origin ?? [0, 0], { discharge } = dem.breach;
        const x0 = dem.breach.x0 - ox, x1 = dem.breach.x1 - ox, y0 = dem.breach.y0 - oy, y1 = dem.breach.y1 - oy;
        if (x0 < 0 || y0 < 0 || x1 > w || y1 > h) throw new RangeError(`${dem.name}: the breach lies outside the crop`);
        const rate = discharge / ((x1 - x0) * (y1 - y0) * dx * dy);
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) source[y * w + x] = rate;
        duration = 3600;
    } else throw new RangeError(`Unknown terrain kind ${kind}`);
    const initial = { state, bed, manning, source };
    if (dem.floodwall) {
        initial.solid = canalFloodwall(dem);
        for (let i = 0; i < n; i++) if (initial.solid[i]) { state.fill(0, 4 * i, 4 * i + 4); source[i] = 0; }
    }
    if (tracer) {
        initial.concentration = new Float32Array(n);
        initial.sourceConcentration = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            if (kind === 'release' && state[4 * i] > 0) initial.concentration[i] = 1;
            if (source[i] > 0) initial.sourceConcentration[i] = 1;
        }
    }
    return { config, initial, duration };
}

// Cases of the terrain test suite (10 m tiles, which are versioned; the 2 m lidar tiles are fetched
// with tools/dem/fetch-3dep.mjs and are not).
export const TERRAIN_CASES = Object.freeze([
    ['yosemite-elcapitan', 'lake'], ['yosemite-elcapitan', 'rain'], ['yosemite-elcapitan', 'release'],
    ['neworleans-lakeview', 'lake'], ['neworleans-lakeview', 'rain'], ['neworleans-lakeview', 'inflow'],
]);
