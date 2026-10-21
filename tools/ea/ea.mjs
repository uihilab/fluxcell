import { compileKernels, createSolver } from '../../src/index.mjs';

// Runs the Environment Agency benchmark cases prepared by tools/ea/prepare.mjs on the fused
// kernel and records water level and speed at the EA output points, peaks, snapshots of the
// flood extent (Test 4), and water budgets.
const status = document.querySelector('#status'), output = document.querySelector('#output');
const CASES = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6a', 'test6b'];
const decode = (text, Type) => { const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0)); return new Type(bytes.buffer); };
const WET = 1e-3; // m; drier output points report the bed level and zero speed

// Exact volume of a piecewise-linear series over [a, b].
function integral(series, a, b) {
    const value = t => {
        if (t <= series[0][0]) return series[0][1];
        for (let k = 1; k < series.length; k++) if (t <= series[k][0]) {
            const [t0, v0] = series[k - 1], [t1, v1] = series[k];
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
        }
        return series.at(-1)[1];
    };
    const knots = [a, ...series.map(s => s[0]).filter(t => t > a && t < b), b];
    let sum = 0;
    for (let k = 1; k < knots.length; k++) sum += (knots[k] - knots[k - 1]) * (value(knots[k - 1]) + value(knots[k])) / 2;
    return sum;
}

async function runCase(device, kernels, name) {
    const spec = await (await fetch(`../../data/ea/cases/${name}.json`)).json();
    const { config } = spec, n = config.width * config.height, area = config.dx * config.dy;
    const state = decode(spec.state, Float32Array), bed = decode(spec.bed, Float32Array), solid = decode(spec.solid, Uint8Array);
    const source = new Float32Array(n);
    const solver = await createSolver(device, config, { state, bed, solid, manning: new Float32Array(n).fill(spec.manning), source }, kernels);
    let volume0 = 0;
    for (let k = 0; k < n; k++) volume0 += state[4 * k] * area;
    const points = spec.outputs.map(p => ({ ...p, bed: bed[p.cell], series: [], peakLevel: -Infinity, peakSpeed: 0, arrival: null }));
    const snapshots = [], started = performance.now();
    const sample = (current, t) => {
        for (const p of points) {
            const h = current[4 * p.cell], wet = h > WET;
            const speed = wet ? Math.hypot(current[4 * p.cell + 1], current[4 * p.cell + 2]) / h : 0;
            const level = wet ? h + p.bed : p.bed;
            p.series.push([+t.toFixed(3), +level.toFixed(5), +speed.toFixed(4)]);
            if (wet) { p.peakLevel = Math.max(p.peakLevel, level); p.peakSpeed = Math.max(p.peakSpeed, speed); p.arrival ??= t; }
        }
    };
    try {
        sample(state, 0);
        const intervals = Math.round(spec.duration / spec.every);
        for (let k = 1; k <= intervals; k++) {
            const t0 = (k - 1) * spec.every, t1 = k * spec.every;
            if (spec.sourceCells) {
                // Interior inflow line: exact interval volume spread over the line's cells.
                const rate = integral(spec.sourceSeries, t0, t1) / spec.every / (spec.sourceCells.length * area);
                for (const c of spec.sourceCells) source[c] = rate;
                solver.setForcing({ source });
            }
            await solver.advance(t1 - solver.time, { batch: 256 });
            const current = await solver.readState();
            sample(current, solver.time);
            if (spec.snapshots?.some(s => Math.abs(s - t1) < 1e-6)) {
                // Flood extent along the centre line y = 1000 m (Test 4): farthest x with h > 0.1 m and 0.2 m.
                const row = Math.floor(1000 / config.dy), reach = threshold => {
                    let x = 0;
                    for (let i = 0; i < config.width; i++) if (current[4 * (row * config.width + i)] > threshold) x = (i + 1) * config.dx;
                    return x;
                };
                snapshots.push({ time: t1, extent10cm: reach(0.1), extent20cm: reach(0.2) });
            }
            if (k % Math.max(1, Math.floor(intervals / 50)) === 0) status.textContent = `${name}: ${(100 * k / intervals).toFixed(0)}%`;
        }
        const final = await solver.readState(), outflow = await solver.readBoundaryOutflow();
        let volume = 0, wetArea = 0;
        for (let k = 0; k < n; k++) { volume += final[4 * k] * area; if (final[4 * k] > WET) wetArea += area; }
        const added = solver.sourceVolume - outflow; // sources plus net boundary inflow
        const residual = volume - volume0 - added;
        return { name, title: spec.title, width: config.width, height: config.height, dx: config.dx,
            openCells: solid.reduce((a, s) => a + 1 - s, 0), simulated: solver.time, wallSeconds: (performance.now() - started) / 1000,
            counters: { ...solver.counters }, volume0, volume, inflowVolume: added, expectedInflowVolume: spec.inflowVolume ?? null,
            relativeBudgetResidual: Math.abs(residual) / Math.max(volume, volume0, 1e-30), wetAreaKm2: wetArea / 1e6, snapshots,
            points: points.map(({ id, x, y, bed, peakLevel, peakSpeed, arrival, series }) =>
                ({ id, x, y, bed, peakLevel: Number.isFinite(peakLevel) ? peakLevel : null, peakSpeed, arrival, final: series.at(-1), series })) };
    } finally { solver.destroy(); }
}

document.querySelector('#run').addEventListener('click', async () => {
    const fine = document.querySelector('#fine').checked;
    const report = { timestamp: new Date().toISOString(), kind: 'ea-benchmarks', userAgent: navigator.userAgent, resolution: fine ? 'half cell size' : 'specified',
        note: 'UK Environment Agency 2D benchmark tests 1-6B (Néelz and Pender 2010; data OGL v3.0), fused MUSCL kernel, f32.', cases: [], errors: [] };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = kernels.sha256;
        for (const name of CASES) {
            try { report.cases.push(await runCase(device, kernels, fine ? `${name}-fine` : name)); }
            catch (error) { report.errors.push(`${name}: ${error.message}`); }
            output.textContent = JSON.stringify(report.cases.map(({ points, ...c }) => c), null, 1);
        }
        device.destroy();
    } catch (error) { report.errors.push(error.message); }
    window.__eaReport = report;
    status.textContent = report.errors.length ? `Done with errors: ${report.errors.join('; ')}` : 'Done.';
});
