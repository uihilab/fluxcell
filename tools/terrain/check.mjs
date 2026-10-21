import { createSolver, compileKernels } from '../../src/webgpu.mjs';
import { MusclReferenceSolver } from '../../src/muscl-reference.mjs';
import { diagnostics } from '../../src/model.mjs';
import { loadDem, terrainScenario, TERRAIN_CASES } from '../../src/terrain.mjs';

const runButton = document.querySelector('#run'), download = document.querySelector('#download');
const status = document.querySelector('#status'), output = document.querySelector('#output');
const tbody = document.querySelector('#table tbody');
let report;

const WET = 0.01; // m, for wetted-area statistics only
function fieldStats(state, config) {
    let wet = 0, maxSpeedWet = 0, maxMomentum = 0;
    for (let k = 0; k < state.length; k += 4) {
        const q = Math.hypot(state[k + 1], state[k + 2]);
        maxMomentum = Math.max(maxMomentum, q);
        if (state[k] > WET) { wet++; maxSpeedWet = Math.max(maxSpeedWet, q / state[k]); }
    }
    // maxSpeed includes films thinner than WET; maxSpeedWet is the speed where h > 1 cm.
    return { ...diagnostics(state, config), maxSpeedWet, maxMomentum, wetAreaKm2: wet * config.dx * config.dy / 1e6 };
}
const maxDifference = (a, b, stride = 1, offset = 0) => {
    let m = 0;
    for (let k = offset; k < a.length; k += stride) m = Math.max(m, Math.abs(a[k] - b[k]));
    return m;
};

async function fullRun(device, kernels, config, initial, duration) {
    const solver = await createSolver(device, config, initial, kernels);
    try {
        const t0 = performance.now();
        const { steps } = await solver.advance(duration, { maxDt: duration, batch: 128 });
        const seconds = (performance.now() - t0) / 1000;
        const state = await solver.readState(), outflow = await solver.readBoundaryOutflow();
        return { state, outflow, steps, seconds, time: solver.time, sourceVolume: solver.sourceVolume,
            counters: { ...solver.counters }, datum: solver.datum };
    } finally { solver.destroy(); }
}

async function stepRun(device, kernels, config, initial, steps, cpu) {
    const solver = await createSolver(device, config, initial, kernels);
    try {
        for (let i = 0; i < steps; i++) { const dt = await solver.step(1); cpu?.step(dt); }
        return await solver.readState();
    } finally { solver.destroy(); }
}

async function runCase(device, kernels, dem, kind) {
    const { config, initial, duration } = terrainScenario(dem, kind);
    const label = `${dem.name} ${kind}`;
    status.textContent = `${label}: fused MUSCL, ${duration} s`;
    const fused = await fullRun(device, kernels['muscl-fused'], config, initial, duration);
    const volume0 = diagnostics(initial.state, config).volume;
    const stats = fieldStats(fused.state, config);
    const residual = stats.volume + fused.outflow - volume0 - fused.sourceVolume;
    // f32 budget error scales with the largest volume term (initial + sources, final, boundary flux).
    const relativeBudgetResidual = Math.abs(residual) / Math.max(volume0 + fused.sourceVolume, stats.volume, Math.abs(fused.outflow), 1e-30);
    const result = { tile: dem.name, kind, duration, config: { ...config }, steps: fused.steps, time: fused.time,
        wallSeconds: fused.seconds, stepsPerSecond: fused.steps / fused.seconds,
        cellStepsPerSecond: fused.steps * config.width * config.height / fused.seconds,
        ...stats, boundaryOutflow: fused.outflow, sourceVolume: fused.sourceVolume, initialVolume: volume0,
        absoluteBudgetResidual: residual, relativeBudgetResidual, counters: fused.counters };
    const checks = { finiteAndNonnegative: stats.minDepth >= 0, massBudget: relativeBudgetResidual <= 1e-5,
        reachedTime: Math.abs(fused.time - duration) <= 1e-5 * duration };

    if (kind === 'lake') {
        result.lake = { maxDepthChange: maxDifference(fused.state, initial.state, 4, 0), maxSpeed: stats.maxSpeed };
        checks.lakeAtRest = result.lake.maxDepthChange <= 1e-5 && stats.maxSpeed <= 1e-4;
        // Diagnostic: the same lake without the f64 datum shift (absolute NAVD88 elevations in f32).
        status.textContent = `${label}: datum diagnostic`;
        const raw = terrainScenario(dem, kind, { datum: 0 });
        const absolute = await fullRun(device, kernels['muscl-fused'], raw.config, raw.initial, raw.duration);
        const absStats = fieldStats(absolute.state, raw.config);
        result.withoutDatumShift = { maxDepthChange: maxDifference(absolute.state, raw.initial.state, 4, 0),
            maxSpeed: absStats.maxSpeed, steps: absolute.steps };
    }

    status.textContent = `${label}: fused vs multi-pass (300 steps)`;
    const [a, b] = [await stepRun(device, kernels['muscl-fused'], config, initial, 300),
        await stepRun(device, kernels['muscl-hancock'], config, initial, 300)];
    result.fusedVsMultiPass = maxDifference(a, b);
    checks.fusedEquivalence = result.fusedVsMultiPass === 0;

    status.textContent = `${label}: f64 CPU reference (40 steps)`;
    const cpu = new MusclReferenceSolver(config, initial);
    const gpu40 = await stepRun(device, kernels['muscl-fused'], config, initial, 40, cpu);
    const qScale = Math.max(1, fieldStats(cpu.state, config).maxMomentum);
    result.cpuAgreement = { steps: 40, maxDepthError: maxDifference(gpu40, cpu.state, 4, 0),
        maxMomentumError: Math.max(maxDifference(gpu40, cpu.state, 4, 1), maxDifference(gpu40, cpu.state, 4, 2)), momentumScale: qScale };
    result.cpuAgreement.maxStateError = result.cpuAgreement.maxDepthError; // table column
    checks.cpuAgreement = result.cpuAgreement.maxDepthError <= 1e-3 && result.cpuAgreement.maxMomentumError <= 1e-3 * qScale;

    if (kind !== 'lake') {
        status.textContent = `${label}: first-order Rusanov comparison`;
        const rus = await fullRun(device, kernels['rusanov'], config, initial, duration);
        const rStats = fieldStats(rus.state, config);
        let l1 = 0;
        for (let k = 0; k < rus.state.length; k += 4) l1 += Math.abs(rus.state[k] - fused.state[k]);
        result.rusanov = { steps: rus.steps, maxDepth: rStats.maxDepth, maxSpeed: rStats.maxSpeed, maxSpeedWet: rStats.maxSpeedWet, wetAreaKm2: rStats.wetAreaKm2,
            boundaryOutflow: rus.outflow, meanAbsDepthDifference: l1 / (config.width * config.height), wallSeconds: rus.seconds };
    }
    result.checks = checks;
    result.passed = Object.values(checks).every(Boolean);
    return result;
}

function addRow(r) {
    const tr = document.createElement('tr');
    for (const v of [r.tile.split('-')[0], r.kind, r.steps, r.relativeBudgetResidual.toExponential(1), r.maxDepth.toFixed(2),
        r.maxSpeedWet.toFixed(2), r.wetAreaKm2.toFixed(3), r.boundaryOutflow.toFixed(0), r.counters.retries,
        r.fusedVsMultiPass === 0 ? 'yes' : r.fusedVsMultiPass.toExponential(1), r.cpuAgreement.maxStateError.toExponential(1),
        r.stepsPerSecond.toFixed(0), r.passed ? 'PASS' : 'FAIL']) {
        const td = document.createElement('td'); td.textContent = v; tr.append(td);
    }
    tbody.append(tr);
}

runButton.addEventListener('click', async () => {
    runButton.disabled = true; download.disabled = true; tbody.replaceChildren();
    report = { timestamp: new Date().toISOString(), kind: 'terrain', userAgent: navigator.userAgent, cases: [], errors: [] };
    let device;
    try {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = {};
        for (const scheme of ['muscl-fused', 'muscl-hancock', 'rusanov']) kernels[scheme] = await compileKernels(device, scheme);
        report.shaderSha256 = Object.fromEntries(Object.entries(kernels).map(([k, v]) => [k, v.sha256]));
        const dems = {};
        for (const [name, kind] of TERRAIN_CASES) {
            dems[name] ??= await loadDem(name);
            report.dems = Object.fromEntries(Object.values(dems).map(d => [d.name, d.meta]));
            try {
                const result = await runCase(device, kernels, dems[name], kind);
                report.cases.push(result); addRow(result);
            } catch (error) { report.errors.push(`${name} ${kind}: ${error.message}`); }
            output.textContent = JSON.stringify(report, null, 2);
        }
        report.passed = report.cases.length === TERRAIN_CASES.length && report.cases.every(c => c.passed) && !report.errors.length;
        status.textContent = report.passed ? 'PASS — terrain checks completed.' : 'FAIL — inspect report.';
    } catch (error) {
        report.errors.push(error.message);
        status.textContent = `FAIL — ${error.message}`;
    } finally {
        device?.destroy();
        output.textContent = JSON.stringify(report, null, 2);
        runButton.disabled = false; download.disabled = false;
    }
});

download.addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = 'terrain-checks.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
