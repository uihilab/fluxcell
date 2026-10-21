import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { loadDem, terrainScenario } from '../../src/terrain.mjs';
import { diagnostics } from '../../src/model.mjs';

// Runs on the 2 m lidar tiles (fused kernel, f32): the 17th Street Canal breach (1024 x 1024 crop,
// one hour, lake water traced) and the Yosemite release (full 1536 x 1536 tile). Fields at selected
// times go to data/figures/lidar-fields.json (not versioned); budgets, retries and wet areas are
// shown on the page (window.__lidar).
const status = document.querySelector('#status'), output = document.querySelector('#output');
const say = text => { status.textContent = text; };
const b64 = array => { let s = ''; const bytes = new Uint8Array(new Float32Array(array).buffer); for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };
const lowFroude = new URLSearchParams(location.search).has('lowFroude');
const CASES = [
    { dem: 'neworleans-17thstreet-2m', kind: 'breach', times: [900, 1800, 3600], wet: 0.01 },
    { dem: 'yosemite-elcapitan-2m', kind: 'release', times: [30, 120, 300], wet: 0.01 },
];

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'lidar', userAgent: navigator.userAgent, lowFroude, cases: [], errors: [] };
    const fields = { timestamp: report.timestamp, cases: [] };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice({ requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = kernels.sha256;
        for (const spec of CASES) {
            const dem = await loadDem(spec.dem), { config, initial } = terrainScenario(dem, spec.kind, { tracer: true });
            const solver = await createSolver(device, { ...config, lowFroude }, initial, kernels), n = config.width * config.height;
            const volume0 = diagnostics(initial.state, solver.config).volume, row = { ...spec, width: config.width, height: config.height,
                dx: config.dx, origin: dem.origin, cells: n, snapshots: [] };
            const stored = { dem: spec.dem, kind: spec.kind, width: config.width, height: config.height, dx: config.dx, origin: dem.origin, times: [] };
            const started = performance.now();
            try {
                for (const t of spec.times) {
                    while (solver.time < t - 1e-6) {
                        await solver.advance(Math.min(60, t - solver.time), { batch: 256 });
                        say(`${spec.dem} ${spec.kind}: t = ${solver.time.toFixed(0)} s`);
                    }
                    const state = await solver.readState(), amount = await solver.readTracer(), outflow = await solver.readBoundaryOutflow();
                    const stats = diagnostics(state, solver.config);
                    const residual = stats.volume + outflow - volume0 - solver.sourceVolume;
                    const depth = new Float32Array(n), speed = new Float32Array(n), conc = new Float32Array(n);
                    let wet = 0, maxDepth = 0, maxSpeed = 0;
                    for (let i = 0; i < n; i++) {
                        const h = state[4 * i];
                        depth[i] = h;
                        speed[i] = h > spec.wet ? Math.hypot(state[4 * i + 1], state[4 * i + 2]) / h : 0;
                        conc[i] = h > 0 ? amount[i] / h : 0;
                        if (h > spec.wet) wet++;
                        maxDepth = Math.max(maxDepth, h); maxSpeed = Math.max(maxSpeed, speed[i]);
                    }
                    row.snapshots.push({ time: solver.time, volume: stats.volume, outflow, sourceVolume: solver.sourceVolume,
                        relativeBudgetResidual: Math.abs(residual) / Math.max(volume0 + solver.sourceVolume, stats.volume, 1e-30),
                        wetAreaKm2: wet * config.dx * config.dy / 1e6, maxDepth, maxSpeed,
                        retries: solver.counters.retries, wallSeconds: (performance.now() - started) / 1000 });
                    stored.times.push({ time: solver.time, depth: b64(depth), speed: b64(speed), concentration: b64(conc) });
                }
            } finally { solver.destroy(); }
            report.cases.push(row); fields.cases.push(stored);
            output.textContent = JSON.stringify(report.cases, null, 1).slice(0, 6000);
        }
        device.destroy();
    } catch (error) { report.errors.push(error.stack ?? error.message); }
    window.__lidar = report; window.__lidarFields = fields;
    say(report.errors.length ? `Errors: ${report.errors.length}` : 'Done.');
});
