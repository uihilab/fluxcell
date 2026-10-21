// Builds solver-ready cases for the UK Environment Agency 2D benchmark tests 1 to 6B (Néelz and
// Pender 2010; data: Open Government Licence v3.0) from the extracted 2010 datasets.
// Usage: node tools/ea/prepare.mjs [data/ea/extracted] [data/ea/cases]
// Each test is prepared at the specified resolution and at half the cell size. Beds are
// area-weighted averages of the DEM over each cell; cells without DEM data or outside the
// active area become solid. Arrays are stored as base64 Float32 or Uint8.
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const [source = 'data/ea/extracted', target = 'data/ea/cases'] = process.argv.slice(2);
const folder = test => { const d = readdirSync(source).find(n => n.startsWith(`Test${test}_`)); return join(source, d, readdirSync(join(source, d))[0]); };
const file = (test, name) => join(folder(test), readdirSync(folder(test)).find(n => n.toLowerCase() === name.toLowerCase()));

function readAsc(path) {
    const text = readFileSync(path, 'utf8').split(/\r?\n/);
    const header = {};
    for (let k = 0; k < 6; k++) { const [key, value] = text[k].trim().split(/\s+/); header[key.toLowerCase()] = Number(value); }
    const values = new Float64Array(header.ncols * header.nrows);
    let n = 0;
    for (let k = 6; k < text.length; k++) for (const token of text[k].trim().split(/\s+/)) if (token) values[n++] = Number(token);
    if (n !== values.length) throw new Error(`${path}: expected ${values.length} values, read ${n}`);
    return { ...header, values };
}

const readCsv = path => readFileSync(path, 'utf8').trim().split(/\r?\n/).slice(1).map(l => l.split(',').map(s => s.trim()));

// First polygon ring of an ESRI shapefile (shape type 5).
function readPolygon(path) {
    const b = readFileSync(path);
    const content = b.subarray(108);
    const parts = content.readInt32LE(36), count = content.readInt32LE(40);
    const start = 44 + 4 * parts, ring = [];
    for (let k = 0; k < count; k++) ring.push([content.readDoubleLE(start + 16 * k), content.readDoubleLE(start + 16 * k + 8)]);
    return ring;
}
function inside([x, y], ring) {
    let c = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i], [xj, yj] = ring[j];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
}

// Area-weighted average of the DEM over each model cell (row 0 = south). NaN where no data.
function resample(dem, x0, y0, d, width, height) {
    const s = dem.cellsize, top = dem.yllcorner + dem.nrows * s, bed = new Float64Array(width * height);
    for (let j = 0; j < height; j++) for (let i = 0; i < width; i++) {
        const xa = x0 + i * d, xb = xa + d, ya = y0 + j * d, yb = ya + d;
        const c0 = Math.max(0, Math.floor((xa - dem.xllcorner) / s)), c1 = Math.min(dem.ncols - 1, Math.ceil((xb - dem.xllcorner) / s) - 1);
        const r0 = Math.max(0, Math.floor((top - yb) / s)), r1 = Math.min(dem.nrows - 1, Math.ceil((top - ya) / s) - 1);
        let sum = 0, area = 0;
        for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
            const v = dem.values[r * dem.ncols + c];
            if (v === dem.nodata_value) continue;
            const cx = dem.xllcorner + c * s, cy = top - (r + 1) * s;
            const w = Math.max(0, Math.min(xb, cx + s) - Math.max(xa, cx)) * Math.max(0, Math.min(yb, cy + s) - Math.max(ya, cy));
            sum += w * v; area += w;
        }
        bed[j * width + i] = area > 0.5 * d * d ? sum / area : NaN;
    }
    return bed;
}

const b64 = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength).toString('base64');
const series = (rows, scaleTime) => rows.map(([t, v]) => [Number(t) * scaleTime, Number(v)]);
const outputs = (test, x0, y0, d, width) => readCsv(file(test, `Test${test}output.csv`)).map(([id, x, y]) => {
    const i = Math.floor((Number(x) - x0) / d), j = Math.floor((Number(y) - y0) / d);
    return { id, x: Number(x), y: Number(y), cell: j * width + i };
});

function finish(name, spec) {
    const { width, height, bed, solid, level, manning } = spec, n = width * height;
    const state = new Float32Array(4 * n), bed32 = new Float32Array(n), solid8 = new Uint8Array(n);
    for (let k = 0; k < n; k++) {
        solid8[k] = solid[k] || !Number.isFinite(bed[k]) ? 1 : 0;
        bed32[k] = Number.isFinite(bed[k]) ? bed[k] : 0;
        if (!solid8[k]) state[4 * k] = Math.max(0, Math.fround(level(k) - bed32[k]));
    }
    const out = { name, ...spec.meta, config: { width, height, dx: spec.d, dy: spec.d, boundaryX: 'wall', boundaryY: 'wall', ...spec.config },
        manning, origin: [spec.x0, spec.y0], state: b64(state), bed: b64(bed32), solid: b64(solid8), outputs: spec.outputs };
    writeFileSync(join(target, `${name}.json`), JSON.stringify(out));
    const open = solid8.reduce((a, s) => a + (1 - s), 0);
    console.log(`${name}: ${width}x${height} at ${spec.d} m, ${open} open cells`);
}

mkdirSync(target, { recursive: true });
for (const fine of [false, true]) {
    const tag = fine ? '-fine' : '';
    // Test 1: stage hydrograph on the west edge of a 700 m x 100 m domain; level 9.7 m initially.
    { const d = fine ? 5 : 10, dem = readAsc(file(1, 'test1DEM.asc')), w = 700 / d, h = 100 / d;
      finish(`test1${tag}`, { width: w, height: h, d, x0: 0, y0: 0, bed: resample(dem, 0, 0, d, w, h), solid: [], manning: 0.03,
        level: () => 9.7, outputs: outputs(1, 0, 0, d, w),
        config: { hydrographs: [{ edge: 'west', kind: 'stage', series: series(readCsv(file(1, 'Test1BC.csv')), 60) }] },
        meta: { title: 'Test 1: flooding a disconnected water body', duration: 72000, every: 60 } }); }
    // Test 2: discharge on the top 100 m of the west edge; dry start; 48 h.
    { const d = fine ? 10 : 20, dem = readAsc(file(2, 'test2DEM.asc')), w = 2000 / d, h = 2000 / d;
      finish(`test2${tag}`, { width: w, height: h, d, x0: 0, y0: 0, bed: resample(dem, 0, 0, d, w, h), solid: [], manning: 0.03,
        level: () => -Infinity, outputs: outputs(2, 0, 0, d, w),
        config: { hydrographs: [{ edge: 'west', start: h - 100 / d, end: h, kind: 'discharge', series: series(readCsv(file(2, 'Test2_BC.csv')), 60) }] },
        meta: { title: 'Test 2: filling of floodplain depressions', duration: 172800, every: 300, inflowVolume: 97200 } }); }
    // Test 3: discharge on the whole west edge (100 m); dry start; 15 min.
    { const d = fine ? 2.5 : 5, dem = readAsc(file(3, 'test3DEM.asc')), w = 300 / d, h = 100 / d;
      finish(`test3${tag}`, { width: w, height: h, d, x0: 0, y0: 0, bed: resample(dem, 0, 0, d, w, h), solid: [], manning: 0.01,
        level: () => -Infinity, outputs: [{ id: '1', x: 150, y: 50 }, { id: '2', x: 250, y: 50 }].map(p => ({ ...p, cell: Math.floor(p.y / d) * w + Math.floor(p.x / d) })),
        config: { hydrographs: [{ edge: 'west', kind: 'discharge', series: series(readCsv(file(3, 'Test3BC.csv')), 1) }] },
        meta: { title: 'Test 3: momentum conservation over a small obstruction', duration: 900, every: 2, inflowVolume: 1310 } }); }
    // Test 4: flat floodplain 1000 m x 2000 m; discharge on 20 m centred on the west edge; 5 h.
    { const d = fine ? 2.5 : 5, w = 1000 / d, h = 2000 / d;
      finish(`test4${tag}`, { width: w, height: h, d, x0: 0, y0: 0, bed: new Float64Array(w * h), solid: [], manning: 0.05,
        level: () => -Infinity, outputs: outputs(4, 0, 0, d, w),
        config: { hydrographs: [{ edge: 'west', start: 990 / d, end: 1010 / d, kind: 'discharge', series: series(readCsv(file(4, 'Test4BC.csv')), 60) }] },
        meta: { title: 'Test 4: speed of flood propagation over an extended floodplain', duration: 18000, every: 20,
            snapshots: [1800, 3600, 7200, 10800, 14400] } }); }
    // Test 5: valley outline from the active-area polygon; inflow along an interior line as a
    // distributed source (time series applied by the runner); 30 h.
    { const d = fine ? 25 : 50, dem = readAsc(file(5, 'Test5DEM.asc')), ring = readPolygon(file(5, 'Test5ActiveArea_region.shp'));
      const x0 = 232315, y0 = 829985, w = Math.ceil((245072.1 - x0) / d), h = Math.ceil((841233.1 - y0) / d);
      const bed = resample(dem, x0, y0, d, w, h), solid = new Uint8Array(w * h);
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) solid[j * w + i] = inside([x0 + (i + 0.5) * d, y0 + (j + 0.5) * d], ring) ? 0 : 1;
      const [a, b] = [[232596.797, 830479.5], [232782.953, 830293.5]], cells = new Set();
      for (let k = 0; k <= 400; k++) {
          const x = a[0] + (b[0] - a[0]) * k / 400, y = a[1] + (b[1] - a[1]) * k / 400;
          const c = Math.floor((y - y0) / d) * w + Math.floor((x - x0) / d);
          if (!solid[c] && Number.isFinite(bed[c])) cells.add(c);
      }
      finish(`test5${tag}`, { width: w, height: h, d, x0, y0, bed, solid, manning: 0.04, level: () => -Infinity,
        outputs: outputs(5, x0, y0, d, w), config: {},
        meta: { title: 'Test 5: valley flooding', duration: 108000, every: 60, inflowVolume: 9450000,
            sourceCells: [...cells], sourceSeries: series(readCsv(file(5, 'Test5BC.csv')), 60) } }); }
    // Tests 6A and 6B: dam break around an isolated building; 10 m DEM cells (dam and building)
    // are solid; still water at 0.4 m / 0.02 m (6A) and 8 m / 0.4 m (6B) either side of x = 0.
    for (const [variant, base, scale, n, duration, every] of [['6a', 0.1, 1, 0.01, 120, 0.1], ['6b', 2, 20, 0.05, 1800, 1]]) {
        const d = fine ? base / 2 : base, dem = readAsc(file(6, `Test6${variant.slice(1).toUpperCase()}DEM.asc`));
        const x0 = dem.xllcorner, y0 = dem.yllcorner, w = Math.floor(dem.ncols * dem.cellsize / d), h = Math.floor(dem.nrows * dem.cellsize / d);
        const bed = resample(dem, x0, y0, d, w, h), solid = bed.map(z => z > 9.5 ? 1 : 0);
        finish(`test${variant}${tag}`, { width: w, height: h, d, x0, y0, bed, solid, manning: n,
          level: k => (x0 + ((k % w) + 0.5) * d < 0 ? 0.4 : 0.02) * scale,
          outputs: readCsv(file(6, `Test6${variant.slice(1).toUpperCase()}output.csv`)).map(([id, x, y]) =>
              ({ id, x: Number(x), y: Number(y), cell: Math.floor((Number(y) - y0) / d) * w + Math.floor((Number(x) - x0) / d) })),
          config: {}, meta: { title: `Test ${variant.toUpperCase()}: dam break around an isolated building`, duration, every } });
    }
}
