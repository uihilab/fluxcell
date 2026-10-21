// Compact data for the viewer demos (run once with Node from the repository root):
//   node tools/nthmp/demo-data.mjs
// 1. NTHMP BP7 (Monai Valley): bed elevation as f32 (rows south to north) and the incident-wave
//    record, from data/nthmp/BP07 (NTHMP benchmark repository; Liu et al., 2008), into
//    examples/viewer/data/.
// 2. The 2 m lidar windows used by the viewer, cut from the full tiles in data/dem (USGS 3DEP,
//    public domain), as data/dem/<name>.demo.f32/json; src/terrain.mjs loadDem falls back to them
//    when the full tile is absent.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { monaiCase } from '../../src/benchmarks.mjs';

const monai = monaiCase(readFileSync('data/nthmp/BP07/Benchmark_2_Bathymetry.txt', 'utf8'),
    readFileSync('data/nthmp/BP07/Benchmark_2_input.txt', 'utf8'));
mkdirSync('examples/viewer/data', { recursive: true });
writeFileSync('examples/viewer/data/monai-bed.f32', Buffer.from(Float32Array.from(monai.initial.bed).buffer));
const wave = monai.config.hydrographs[0];
writeFileSync('examples/viewer/data/monai.json', JSON.stringify({
    width: monai.config.width, height: monai.config.height, dx: monai.config.dx, still: wave.still, incident: wave.series,
    source: 'NTHMP benchmark problem 7 (github.com/rjleveque/nthmp-benchmark-problems, BP07-DmitryN-Monai_valley_beach): '
        + 'Benchmark_2_Bathymetry.txt (still-water depth, here bed elevation = -depth) and Benchmark_2_input.txt (incident water level), '
        + 'followed by one point at still water 0.05 s after the record ends. Laboratory model of the Monai Valley coast '
        + '(Central Research Institute for Electric Power Industry; Liu, Yeh and Synolakis, 2008).',
}));

for (const [name, crop] of [['yosemite-elcapitan-2m', [404, 700, 768, 768]], ['neworleans-17thstreet-2m', [640, 1084, 640, 640]]]) {
    const meta = JSON.parse(readFileSync(`data/dem/${name}.json`, 'utf8'));
    const bytes = readFileSync(`data/dem/${name}.f32`), full = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const [x0, y0, w, h] = crop, W = meta.width, H = meta.height, out = new Float32Array(w * h);
    // Rows stay north to south, as in the full tile; row 0 of the window is its northern row.
    for (let r = 0; r < h; r++) out.set(full.subarray((H - 1 - (y0 + h - 1 - r)) * W + x0, (H - 1 - (y0 + h - 1 - r)) * W + x0 + w), r * w);
    writeFileSync(`data/dem/${name}.demo.f32`, Buffer.from(out.buffer));
    writeFileSync(`data/dem/${name}.demo.json`, JSON.stringify({ ...meta, width: w, height: h, window: { of: name, origin: [x0, y0], width: W, height: H },
        note: `Window [${crop}] (x0, y0 from the south-west, width, height) of ${name}, for the viewer demos.` }, null, 2));
}
console.log('wrote examples/viewer/data/monai-bed.f32, monai.json and data/dem/*.demo.*');
