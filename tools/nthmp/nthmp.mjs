import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { CONICAL_ISLAND, conicalIslandCase, MONAI, monaiCase } from '../../src/benchmarks.mjs';

// NTHMP BP6 on the fused kernel. Each run ("case:dx:manning"; "monai:manning" for BP7, below) records the water level eta = h + z - d
// at the gauges every 0.04 s (the laboratory sampling interval) and, at the same interval, the
// maximum depth of every cell within 3.9 m of the island centre. Runup is the bed elevation, above
// still water, of the highest cell along a ray from the island centre whose maximum depth exceeded
// a wetting threshold. The results are compared with the laboratory records offline.
const status = document.querySelector('#status'), output = document.querySelector('#output');
const say = text => { status.textContent = text; };
const b64 = array => { let s = ''; const bytes = new Uint8Array(new Float32Array(array).buffer); for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };
const SAMPLE = 0.04, DURATION = 22, THRESHOLDS = [0.0005, 0.001, 0.002];

async function read(device, solver, ranges) { // ranges: [[firstCell, count], ...] of the state buffer
    const total = ranges.reduce((a, r) => a + r[1], 0);
    const buffer = device.createBuffer({ size: 16 * total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    let at = 0;
    for (const [first, count] of ranges) { encoder.copyBufferToBuffer(solver.currentState, 16 * first, buffer, 16 * at, 16 * count); at += count; }
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.destroy();
    return values;
}

// Runup above still water along rays at every 0.5 degree; angle 0 points to -y and 90 to +x (the lee
// side), the convention of the gauge transects in the NTHMP description.
function runup(maxDepth, bed, box, dx, threshold) {
    const p = CONICAL_ISLAND, out = [];
    for (let a = 0; a < 360; a += 0.5) {
        const t = a * Math.PI / 180, ux = Math.sin(t), uy = -Math.cos(t);
        let best = -Infinity;
        for (let r = p.crestRadius; r <= p.toeRadius + 0.2; r += dx / 2) {
            const i = Math.floor((p.centre[0] + r * ux) / dx) - box.i0, j = Math.floor((p.centre[1] + r * uy) / dx) - box.j0;
            const k = j * box.w + i;
            if (maxDepth[k] > threshold) best = Math.max(best, bed[k] - p.depth);
        }
        out.push(best);
    }
    return out;
}

async function run(device, kernels, kind, dx, manning, measured = false) {
    const c = conicalIslandCase(dx, kind, { manning, measured }), W = c.config.width, p = CONICAL_ISLAND;
    const reach = p.toeRadius + 0.3;
    const box = { i0: Math.floor((p.centre[0] - reach) / dx), j0: Math.floor((p.centre[1] - reach) / dx) };
    box.w = Math.ceil(2 * reach / dx); box.h = box.w;
    const bed = new Float32Array(box.w * box.h), maxDepth = new Float32Array(box.w * box.h);
    for (let j = 0; j < box.h; j++) for (let i = 0; i < box.w; i++) bed[j * box.w + i] = c.initial.bed[(box.j0 + j) * W + box.i0 + i];
    const ids = Object.keys(c.gauges), cells = ids.map(g => c.gauges[g]);
    const rows = Array.from({ length: box.h }, (_, j) => [(box.j0 + j) * W + box.i0, box.w]);
    const gpu = await createSolver(device, c.config, c.initial, kernels);
    const series = Object.fromEntries(ids.map(g => [g, []])), times = [], snapshots = [];
    const wall = performance.now();
    try {
        for (let n = 0; n * SAMPLE <= DURATION + 1e-9; n++) {
            if (n > 0) await gpu.advance(n * SAMPLE - gpu.time, { batch: 256 });
            const v = await read(device, gpu, [...cells.map(k => [k, 1]), ...rows]);
            times.push(gpu.time);
            ids.forEach((g, m) => series[g].push(v[4 * m] + c.initial.bed[cells[m]] - p.depth));
            for (let q = 0, o = 4 * cells.length; q < maxDepth.length; q++, o += 4) if (v[o] > maxDepth[q]) maxDepth[q] = v[o];
            if (dx === 0.025 && manning === 0 && [2, 4, 6, 8].includes(Math.round(gpu.time * 100) / 100)) {
                const s = await gpu.readState(), eta = new Float32Array(s.length / 4);
                for (let q = 0; q < eta.length; q++) eta[q] = s[4 * q] > 1e-4 ? s[4 * q] + c.initial.bed[q] - p.depth : NaN;
                snapshots.push({ t: gpu.time, eta: b64(eta) });
            }
            if (n % 25 === 0) say(`case ${kind}, dx = ${dx} m, n = ${manning}: t = ${gpu.time.toFixed(2)} s`);
        }
    } finally { gpu.destroy(); }
    return { kind, measured, dx, manning, width: W, height: c.config.height, H: c.H, x0: c.x0, celerity: c.celerity,
        wallSeconds: (performance.now() - wall) / 1000, counters: gpu.counters, times, series, box,
        maxDepth: b64(maxDepth), bed: b64(bed),
        runup: Object.fromEntries(THRESHOLDS.map(h => [h, runup(maxDepth, bed, box, dx, h)])), snapshots };
}

// NTHMP BP7 (Monai Valley): water level at gauges 5, 7 and 9 every 0.05 s (the laboratory interval) to
// 25 s, the maximum depth of every cell in the valley region (x 4.6 to 5.45 m, y 1.3 to 2.6 m) at the
// same interval, and the depth there at the times of video frames 10, 25, 40, 55 and 70 (15.3 s and
// every 0.5 s after, as given in the description).
async function runMonai(device, kernels, manning) {
    const [bathymetry, input] = await Promise.all(['Benchmark_2_Bathymetry.txt', 'Benchmark_2_input.txt']
        .map(f => fetch(`../../data/nthmp/BP07/${f}`).then(r => { if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r.text(); })));
    const c = monaiCase(bathymetry, input, { manning }), W = c.config.width, dx = MONAI.dx;
    const box = { i0: Math.floor(4.6 / dx), j0: Math.floor(1.3 / dx) };
    box.w = Math.ceil(0.85 / dx); box.h = Math.ceil(1.3 / dx);
    const bed = new Float32Array(box.w * box.h), maxDepth = new Float32Array(box.w * box.h);
    for (let j = 0; j < box.h; j++) for (let i = 0; i < box.w; i++) bed[j * box.w + i] = c.initial.bed[(box.j0 + j) * W + box.i0 + i];
    const ids = Object.keys(c.gauges), cells = ids.map(g => c.gauges[g]);
    const rows = Array.from({ length: box.h }, (_, j) => [(box.j0 + j) * W + box.i0, box.w]);
    const gpu = await createSolver(device, c.config, c.initial, kernels);
    const series = Object.fromEntries(ids.map(g => [g, []])), times = [], frames = [];
    const frameTimes = [15.3, 15.8, 16.3, 16.8, 17.3];
    const wall = performance.now();
    try {
        const sampleTimes = [];
        for (let n = 0; n * 0.05 <= 25 + 1e-9; n++) sampleTimes.push(n * 0.05);
        const all = [...new Set([...sampleTimes, ...frameTimes].map(t => Math.round(t * 1000) / 1000))].sort((a, b) => a - b);
        for (const t of all) {
            if (t > 0) await gpu.advance(t - gpu.time, { batch: 256 });
            const v = await read(device, gpu, [...cells.map(k => [k, 1]), ...rows]);
            const depth = new Float32Array(box.w * box.h);
            for (let q = 0, o = 4 * cells.length; q < maxDepth.length; q++, o += 4) { depth[q] = v[o]; if (v[o] > maxDepth[q]) maxDepth[q] = v[o]; }
            if (frameTimes.some(f => Math.abs(f - t) < 1e-6)) frames.push({ t: gpu.time, depth: b64(depth) });
            if (sampleTimes.some(f => Math.abs(f - t) < 1e-6)) {
                times.push(gpu.time);
                ids.forEach((g, m) => series[g].push(v[4 * m] + c.initial.bed[cells[m]]));
            }
            if (Math.round(t * 20) % 50 === 0) say(`Monai, n = ${manning}: t = ${gpu.time.toFixed(2)} s`);
        }
    } finally { gpu.destroy(); }
    return { kind: 'monai', dx, manning, width: W, height: c.config.height, wallSeconds: (performance.now() - wall) / 1000,
        counters: gpu.counters, times, series, box, maxDepth: b64(maxDepth), bed: b64(bed), frames };
}

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'nthmp', userAgent: navigator.userAgent, runs: [], errors: [] };
    try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, 'muscl-fused');
        for (const spec of document.querySelector('#runs').value.split(',')) {
            const [kind, dx, manning] = spec.split(':');
            report.runs.push(kind === 'monai' ? await runMonai(device, kernels, Number(dx))
                : await run(device, kernels, kind[0], Number(dx), Number(manning), kind.endsWith('m')));
        }
        device.destroy();
    } catch (error) { report.errors.push(error.stack ?? error.message); }
    // The report (with fields) is archived by the local server in data/figures/ (not versioned).
    try {
        const name = report.runs.some(r => r.kind === 'monai') ? 'nthmp-bp7' : 'nthmp-bp6';
        const put = await fetch(`../../data/figures/${name}.json`, { method: 'PUT', body: JSON.stringify(report) });
        if (!put.ok) report.errors.push(`archive failed: HTTP ${put.status}`);
    } catch (error) { report.errors.push(`archive failed: ${error.message}`); }
    const brief = { errors: report.errors, runs: report.runs.map(r => ({ kind: r.kind, dx: r.dx, manning: r.manning, cells: r.width * r.height, wallSeconds: r.wallSeconds, counters: r.counters })) };
    window.__nthmp = brief;
    output.textContent = JSON.stringify(brief, null, 2);
    say(report.errors.length ? `Errors: ${report.errors.length}` : 'Done.');
});
