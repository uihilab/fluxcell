import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { diagnostics } from '../../src/model.mjs';
import { fuzzScenario } from '../../src/fuzz.mjs';
// ?lowFroude runs every case with the low-Froude correction (MUSCL schemes only).
const variant = new URLSearchParams(location.search).has('lowFroude') ? c => ({ ...c, lowFroude: true }) : c => c;


const status = document.querySelector('#status'), output = document.querySelector('#output');
const value = id => Number(document.querySelector(id).value);

function check(state, config) {
    let invalid = 0;
    for (let k = 0; k < state.length; k += 4) {
        if (!(state[k] >= 0) || !Number.isFinite(state[k + 1]) || !Number.isFinite(state[k + 2])) invalid++;
    }
    return invalid;
}

async function runCase(device, kernels, seed, steps) {
    const c = fuzzScenario(seed, { steps });
    const result = { seed, description: c.description };
    const volume0 = diagnostics(c.initial.state, { ...c.config, dx: c.config.dx, dy: c.config.dy }).volume;
    let solver;
    try {
        solver = await createSolver(device, variant(c.config), c.initial, kernels['muscl-fused']);
        const t0 = performance.now();
        const { steps: taken } = await solver.advance(1e7, { maxDt: 10, batch: 64, maxSteps: steps });
        result.wallMs = performance.now() - t0;
        const state = await solver.readState(), outflow = await solver.readBoundaryOutflow();
        result.steps = taken; result.time = solver.time; result.counters = { ...solver.counters };
        result.invalidCells = check(state, c.config);
        let volume = 0;
        for (let k = 0; k < state.length; k += 4) volume += state[k] * c.config.dx * c.config.dy;
        const residual = volume + outflow - volume0 - solver.sourceVolume;
        // f32 budget error scales with the largest volume term (initial + sources, final, boundary flux).
        result.relativeBudgetResidual = Math.abs(residual) / Math.max(volume0 + solver.sourceVolume, volume, Math.abs(outflow), 1e-30);
        result.absoluteMeanDepthResidual = Math.abs(residual) / (c.config.width * c.config.height * c.config.dx * c.config.dy);
        result.passed = result.invalidCells === 0
            && (result.relativeBudgetResidual <= 1e-5 || result.absoluteMeanDepthResidual <= 1e-8);
    } catch (error) {
        result.fatal = error.message; result.passed = false;
        if (solver) result.counters = { ...solver.counters };
    } finally { solver?.destroy?.(); }

    if (seed % 10 === 0 && !result.fatal) {
        let a, b;
        try {
            a = await createSolver(device, variant(c.config), c.initial, kernels['muscl-fused']);
            b = await createSolver(device, variant(c.config), c.initial, kernels['muscl-hancock']);
            for (let i = 0; i < 100; i++) { await a.step(10); await b.step(10); }
            const [sa, sb] = [await a.readState(), await b.readState()];
            let diff = 0;
            for (let k = 0; k < sa.length; k++) diff = Math.max(diff, Math.abs(sa[k] - sb[k]));
            result.equivalenceMaxDifference = diff;
            if (diff !== 0) result.passed = false;
        } catch (error) { result.equivalenceFatal = error.message; result.passed = false; }
        finally { a?.destroy?.(); b?.destroy?.(); }
    }
    return result;
}

document.querySelector('#run').addEventListener('click', async () => {
    const first = value('#seed'), count = value('#count'), steps = value('#steps');
    const report = { timestamp: new Date().toISOString(), kind: 'stress', userAgent: navigator.userAgent,
        firstSeed: first, count, steps, cases: [], errors: [] };
    const adapter = await navigator.gpu.requestAdapter();
    report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
    const kernels = {};
    for (const scheme of ['muscl-fused', 'muscl-hancock']) kernels[scheme] = await compileKernels(device, scheme);
    report.shaderSha256 = Object.fromEntries(Object.entries(kernels).map(([k, v]) => [k, v.sha256]));
    const list = document.querySelector('#list').value.split(',').map(Number).filter(Number.isFinite).filter(v => v > 0);
    const seeds = list.length ? list : Array.from({ length: count }, (_, i) => first + i);
    report.seeds = list.length ? list : undefined;
    for (const [index, seed] of seeds.entries()) {
        status.textContent = `seed ${seed} (${index + 1}/${seeds.length})`;
        report.cases.push(await runCase(device, kernels, seed, steps));
        const failed = report.cases.filter(r => !r.passed);
        output.textContent = JSON.stringify({ failed: failed.length, failures: failed.slice(0, 20) }, null, 2);
    }
    const retries = report.cases.map(r => r.counters?.retries ?? 0);
    report.summary = { cases: report.cases.length, failed: report.cases.filter(r => !r.passed).length,
        fatal: report.cases.filter(r => r.fatal).length, casesWithRetries: retries.filter(r => r > 0).length,
        totalRetries: retries.reduce((a, b) => a + b, 0),
        totalSteps: report.cases.reduce((a, r) => a + (r.steps ?? 0), 0),
        maxRelativeBudgetResidual: Math.max(...report.cases.map(r => r.relativeBudgetResidual ?? 0)),
        equivalenceRuns: report.cases.filter(r => 'equivalenceMaxDifference' in r).length };
    report.passed = report.summary.failed === 0 && report.errors.length === 0;
    device.destroy();
    window.__stressReport = report;
    status.textContent = report.passed ? 'PASS' : 'FAIL — inspect report';
    output.textContent = JSON.stringify({ summary: report.summary, failures: report.cases.filter(r => !r.passed).slice(0, 30) }, null, 2);
});
