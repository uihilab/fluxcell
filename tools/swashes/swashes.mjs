import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { ReferenceSolver } from '../../src/reference.mjs';
import { MusclReferenceSolver } from '../../src/muscl-reference.mjs';
import { thackerScenario, thackerAverages, bumpScenario, bumpAverages, bumpSolution, BUMP_CASES } from '../../src/benchmarks.mjs';
import { diagnostics } from '../../src/model.mjs';
// ?lowFroude runs every case with the low-Froude correction (MUSCL schemes only).
const variant = new URLSearchParams(location.search).has('lowFroude') ? c => ({ ...c, lowFroude: true }) : c => c;


const status = document.querySelector('#status'), output = document.querySelector('#output');
const THACKER_GPU = [50, 100, 200, 400], THACKER_CPU = [50, 100];
const BUMP_GPU = [100, 200, 400, 800], BUMP_CPU = [100, 200];

// Mean (L1), root-mean-square (L2) and maximum (Linf) depth errors over all cells.
function errors(h, exact) {
    let l1 = 0, l2 = 0, linf = 0;
    for (let i = 0; i < exact.length; i++) { const e = Math.abs(h[i] - exact[i]); l1 += e; l2 += e * e; linf = Math.max(linf, e); }
    return { l1: l1 / exact.length, l2: Math.sqrt(l2 / exact.length), linf };
}
const depths = state => Float64Array.from({ length: state.length / 4 }, (_, i) => state[4 * i]);
const orders = rows => rows.slice(1).map((r, i) => Math.log2(rows[i].l1 / r.l1));

// Solver adapters: GPU solvers advance in batches; CPU references step to the target time.
async function gpuRunner(device, kernels, config, initial) {
    const solver = await createSolver(device, variant(config), initial, kernels);
    return { solver, advanceTo: async t => { if (t > solver.time) await solver.advance(t - solver.time, { batch: 256 }); },
        state: () => solver.readState(), outflow: () => solver.readBoundaryOutflow(), destroy: () => solver.destroy() };
}
function cpuRunner(Solver, config, initial) {
    const solver = new Solver(Solver === ReferenceSolver ? config : variant(config), initial);
    return { solver, advanceTo: async t => {
        let k = 0;
        while (t - solver.time > 1e-12 * Math.max(1, t)) { solver.step(t - solver.time); if (++k % 2000 === 0) await new Promise(r => setTimeout(r)); }
    }, state: async () => solver.state, outflow: async () => solver.boundaryOutflow, destroy: () => {} };
}

async function thacker(make, label, n) {
    const { config, initial, period } = thackerScenario(n), runner = await make(config, initial);
    const area = config.dx * config.dy, volume0 = diagnostics(initial.state, config).volume, started = performance.now();
    try {
        const rows = [];
        for (const fraction of [0.5, 1, 2]) {
            status.textContent = `${label}: Thacker ${n}², t = ${fraction} T`;
            await runner.advanceTo(fraction * period);
            const state = await runner.state(), exact = thackerAverages(n, runner.solver.time).state;
            let volume = 0;
            for (let k = 0; k < state.length; k += 4) volume += state[k] * area;
            rows.push({ periods: fraction, time: runner.solver.time, ...errors(depths(state), depths(exact)),
                relativeVolumeError: Math.abs(volume - volume0) / volume0 });
        }
        return { n, rows, counters: { ...runner.solver.counters }, wallSeconds: (performance.now() - started) / 1000 };
    } finally { runner.destroy(); }
}

async function bump(make, label, kind, n) {
    const { config, initial, finalTime } = bumpScenario(kind, n), runner = await make(config, initial);
    const started = performance.now();
    try {
        status.textContent = `${label}: bump ${kind}, ${n} cells`;
        await runner.advanceTo(finalTime);
        const state = await runner.state(), exact = bumpAverages(kind, n), q = BUMP_CASES[kind].q;
        let qError = 0;
        for (let i = 0; i < n; i++) qError = Math.max(qError, Math.abs(state[4 * i + 1] - q) / q);
        return { n, time: runner.solver.time, ...errors(depths(state), exact.h), maxRelativeCellDischargeError: qError,
            counters: { ...runner.solver.counters }, wallSeconds: (performance.now() - started) / 1000 };
    } finally { runner.destroy(); }
}

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'swashes', userAgent: navigator.userAgent, errors: [],
        note: 'Depth errors are cell-average errors against exact cell averages (4x4 Gauss). GPU runs are f32; CPU references are f64.',
        thacker: {}, bump: {}, jump: bumpSolution('shock').jump };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const fused = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = fused.sha256;
        const runs = [
            ['muscl-fused (GPU f32)', (c, i) => gpuRunner(device, fused, c, i), THACKER_GPU, BUMP_GPU],
            ['MUSCL (CPU f64)', (c, i) => cpuRunner(MusclReferenceSolver, c, i), THACKER_CPU, BUMP_CPU],
            ['Rusanov (CPU f64)', (c, i) => cpuRunner(ReferenceSolver, c, i), THACKER_CPU, BUMP_CPU],
        ];
        // ?gpuOnly skips the f64 references (for example to compare GPU variants).
        if (new URLSearchParams(location.search).has('gpuOnly')) runs.splice(1);
        report.lowFroude = new URLSearchParams(location.search).has('lowFroude');
        for (const [label, make, thackerSizes, bumpSizes] of runs) {
            const rows = [];
            for (const n of thackerSizes) { rows.push(await thacker(make, label, n)); output.textContent = JSON.stringify(report, null, 1); }
            const atPeriod = rows.map(r => r.rows[1]);
            report.thacker[label] = { rows, ordersAtT: orders(atPeriod) };
            report.bump[label] = {};
            for (const kind of Object.keys(BUMP_CASES)) {
                const bumpRows = [];
                for (const n of bumpSizes) bumpRows.push(await bump(make, label, kind, n));
                report.bump[label][kind] = { rows: bumpRows, orders: orders(bumpRows) };
                output.textContent = JSON.stringify(report, null, 1);
            }
        }
        device.destroy();
    } catch (error) { report.errors.push(error.message); }
    report.passed = report.errors.length === 0;
    window.__swashesReport = report;
    output.textContent = JSON.stringify(report, null, 1);
    status.textContent = report.passed ? 'Done.' : 'FAIL: see errors.';
});
