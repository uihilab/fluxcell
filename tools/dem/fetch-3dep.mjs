// Fetch a square USGS 3DEP DEM tile as raw float32 (no dependencies).
// Usage: node tools/dem/fetch-3dep.mjs <name> <lon> <lat> [cells=256] [dx=10] [survey]
// Writes data/dem/<name>.f32 (row-major, north row first as served) and <name>.json.
// survey: optional raster name pattern (for example LA_2021GreaterNewOrleans or CA_YosemiteNP_2019) that
// locks the mosaic to that lidar survey's rasters ('|' separates several); cells must then be fully covered.
import { writeFile, mkdir } from 'node:fs/promises';

const SERVICE = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage';
const [name, lon, lat, cellsArg = '256', dxArg = '10', survey] = process.argv.slice(2);
if (!name || !Number.isFinite(+lon) || !Number.isFinite(+lat)) {
    console.error('Usage: node tools/dem/fetch-3dep.mjs <name> <lon> <lat> [cells] [dx]');
    process.exit(1);
}
const cells = Number(cellsArg), dx = Number(dxArg), half = cells * dx / 2;
const query = params => `${SERVICE}?${new URLSearchParams({ ...params, f: params.f ?? 'image' })}`;

// 1. Locate the centre in EPSG:5070 (CONUS Albers, metres) from a tiny geographic box.
const probe = await (await fetch(query({ bbox: `${+lon - 1e-4},${+lat - 1e-4},${+lon + 1e-4},${+lat + 1e-4}`,
    bboxSR: 4326, imageSR: 5070, size: '2,2', format: 'bsq', pixelType: 'F32', f: 'json' }))).json();
if (!probe.extent) throw new Error(`Unexpected probe response: ${JSON.stringify(probe).slice(0, 300)}`);
const cx = (probe.extent.xmin + probe.extent.xmax) / 2, cy = (probe.extent.ymin + probe.extent.ymax) / 2;

// 2. Request an exactly square grid with dx-metre cells in that projection.
const bbox = [cx - half, cy - half, cx + half, cy + half], n = cells * cells;
const request = mosaicRule => query({ bbox: bbox.join(','), bboxSR: 5070, imageSR: 5070, size: `${cells},${cells}`,
    format: 'bsq', pixelType: 'F32', noData: -9999, interpolation: 'RSP_BilinearInterpolation',
    ...(mosaicRule ? { mosaicRule: JSON.stringify(mosaicRule) } : {}) });

async function download(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // BSQ payload: cells² float32 values followed by a 1-bit-per-pixel validity mask.
    const payload = new Uint8Array(await response.arrayBuffer());
    if (payload.length !== n * 4 + n / 8) throw new Error(`Expected ${n * 4 + n / 8} bytes, received ${payload.length}`);
    if (payload.subarray(n * 4).some(b => b !== 0xff)) throw new Error('Validity mask reports no-data cells; choose another area');
    const bytes = payload.slice(0, n * 4), values = new Float32Array(bytes.buffer, 0, n);
    let min = Infinity, max = -Infinity, sentinels = 0;
    for (const z of values) {
        if (!Number.isFinite(z) || z <= -9999) throw new Error('DEM contains no-data cells; choose another area');
        if (z !== 0 && Math.abs(z) < 1e-30) sentinels++;
        min = Math.min(min, z); max = Math.max(max, z);
    }
    return { bytes, min, max, sentinels };
}

// The default mosaic has returned FLT_MIN in place of negative (below-datum) elevations at
// <=10 m in New Orleans (observed in 2026) while the validity mask reported valid data.
// Fall back to the covering 1/3 arc-second (~10 m) seamless 3DEP product, locked explicitly.
async function rasters() {
    const catalog = await (await fetch(`${SERVICE.replace('exportImage', 'query')}?${new URLSearchParams({
        geometry: bbox.join(','), geometryType: 'esriGeometryEnvelope', inSR: 5070, spatialRel: 'esriSpatialRelIntersects',
        outFields: 'OBJECTID,Name,LowPS,Category', returnGeometry: false, f: 'json' })}`)).json();
    return (catalog.features ?? []).map(f => f.attributes);
}

let url, result, sourceNote = 'default mosaic';
if (survey) {
    // Lock to one lidar survey: exact provenance, no mixing with coarser products.
    const ids = (await rasters()).filter(a => a.Category === 1 && survey.split('|').some(s => a.Name.includes(s)));
    if (!ids.length) throw new Error(`No 3DEP raster named like ${survey} intersects the tile`);
    url = request({ mosaicMethod: 'esriMosaicLockRaster', lockRasterIds: ids.map(a => a.OBJECTID) });
    result = await download(url);
    if (result.sentinels) throw new Error(`${result.sentinels} FLT_MIN sentinels in the ${survey} rasters`);
    sourceNote = `locked to lidar survey ${survey}: ${ids.map(a => `${a.Name} (${a.OBJECTID}, ${a.LowPS} m)`).join(', ')}`;
} else {
    url = request(); result = await download(url);
}
if (!survey && result.sentinels) {
    const catalog = await (await fetch(`${SERVICE.replace('exportImage', 'query')}?${new URLSearchParams({
        geometry: bbox.join(','), geometryType: 'esriGeometryEnvelope', inSR: 5070, spatialRel: 'esriSpatialRelIntersects',
        outFields: 'OBJECTID,Name,LowPS,Category', returnGeometry: false, f: 'json' })}`)).json();
    const third = (catalog.features ?? []).map(f => f.attributes)
        .filter(a => a.Category === 1 && /^n\d{2}w\d{3}$/.test(a.Name) && a.LowPS > 5 && a.LowPS < 15);
    if (!third.length) throw new Error(`${result.sentinels} FLT_MIN sentinels and no 1/3 arc-second raster to fall back to`);
    url = request({ mosaicMethod: 'esriMosaicLockRaster', lockRasterIds: third.map(a => a.OBJECTID) });
    result = await download(url);
    if (result.sentinels) throw new Error('1/3 arc-second fallback also contains FLT_MIN sentinels');
    sourceNote = `locked to 1/3 arc-second rasters ${third.map(a => `${a.Name} (${a.OBJECTID})`).join(', ')}; default mosaic clamped negatives`;
}
const { bytes, min, max } = result;
await mkdir('data/dem', { recursive: true });
await writeFile(`data/dem/${name}.f32`, bytes);
const meta = { name, source: 'USGS 3DEP ImageServer exportImage (bilinear, F32)', mosaic: sourceNote, url, retrieved: new Date().toISOString(),
    centre: { lon: +lon, lat: +lat }, crs: 'EPSG:5070', bbox, width: cells, height: cells, dx, dy: dx,
    rowOrder: 'north-to-south', units: 'm (NAVD88)', min, max };
await writeFile(`data/dem/${name}.json`, JSON.stringify(meta, null, 2) + '\n');
console.log(`${name}: ${cells}x${cells} @ ${dx} m, z ${min.toFixed(2)}..${max.toFixed(2)} m`);
