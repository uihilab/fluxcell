import { compileKernels, createSolver } from '../../src/index.mjs';
import { loadDem, terrainScenario } from '../../src/terrain.mjs';

// Depth and speed fields of the real-terrain cases at selected times (for the real-terrain figure), stored as
// base64 Float32 arrays (row 0 = south).
const status = document.querySelector('#status');
const CASES = [
    { dem: 'yosemite-elcapitan', kind: 'release', times: [60, 300] },
    { dem: 'yosemite-elcapitan', kind: 'rain', times: [900] },
    { dem: 'neworleans-lakeview', kind: 'inflow', times: [600, 1800] },
];
const b64 = array => { let s = ''; const bytes = new Uint8Array(array.buffer); for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'terrain-snapshots', userAgent: navigator.userAgent, cases: [], errors: [] };
    try {
        const device = await (await navigator.gpu.requestAdapter()).requestDevice();
        const kernels = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = kernels.sha256;
        for (const spec of CASES) {
            const dem = await loadDem(spec.dem), { config, initial } = terrainScenario(dem, spec.kind);
            const solver = await createSolver(device, config, initial, kernels), n = config.width * config.height;
            const fields = [];
            try {
                for (const t of spec.times) {
                    status.textContent = `${spec.dem} ${spec.kind} t = ${t} s`;
                    await solver.advance(t - solver.time, { batch: 256 });
                    const state = await solver.readState(), depth = new Float32Array(n), speed = new Float32Array(n);
                    for (let i = 0; i < n; i++) {
                        const h = state[4 * i]; depth[i] = h;
                        speed[i] = h > 1e-3 ? Math.hypot(state[4 * i + 1], state[4 * i + 2]) / h : 0;
                    }
                    fields.push({ time: solver.time, depth: b64(depth), speed: b64(speed) });
                }
            } finally { solver.destroy(); }
            report.cases.push({ dem: spec.dem, kind: spec.kind, width: config.width, height: config.height, dx: config.dx, fields });
        }
        device.destroy();
    } catch (error) { report.errors.push(error.message); }
    window.__snapshots = report;
    status.textContent = report.errors.length ? `Errors: ${report.errors.join('; ')}` : 'Done.';
});
