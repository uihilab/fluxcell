import { createSolver, compileKernels } from '../../src/webgpu.mjs';
import { ReferenceSolver } from '../../src/reference.mjs';
import { MusclReferenceSolver } from '../../src/muscl-reference.mjs';
import { exactScenario, cellAverages, depthErrors, diagonalScenario, diagonalAverages, steadyBedScenario } from '../../src/analytical.mjs';
import { diagnostics, windStress, sourceDischarge } from "../../src/model.mjs";
import { scenario, frictionExact } from '../../src/scenarios.mjs';
// ?lowFroude runs every case with the low-Froude correction (MUSCL schemes only).
const lowFroude = new URLSearchParams(location.search).has('lowFroude');
const variant = (c, scheme) => (lowFroude && scheme !== 'rusanov' ? { ...c, lowFroude: true } : c);


const run = document.querySelector('#run'), download = document.querySelector('#download');
const status = document.querySelector('#status'), output = document.querySelector('#output');
const method = document.querySelector('#scheme');
let report;

const cpuFor = scheme => scheme === 'rusanov' ? ReferenceSolver : MusclReferenceSolver;
const fused = kernels => kernels.scheme === 'muscl-fused';
// One tolerance for all cases. The thin-film 'sources' case needed 2e-3 with the absolute
// eta = h + z reconstruction; the difference form agrees to ~3e-6.
const referenceTolerance = () => 2e-4;
const EQUILIBRIA = ['constant', 'lake', 'lake-rounded', 'shoreline', 'lake-obstacle', 'open-uniform', 'stage-lake', 'wave-lake'];

// Advance the GPU by `steps` accepted steps (fused: in batches) and replay each dt on the CPU.
async function advanceBoth(gpu, cpu, kernels, steps, maxDt, label) {
    let done = 0;
    while (done < steps) {
        const { dts } = fused(kernels)
            ? await gpu.advance(maxDt * (steps - done), { maxDt, batch: 64, maxSteps: steps - done })
            : { dts: [await gpu.step(maxDt)] };
        for (const dt of dts) cpu.step(dt);
        done += dts.length;
        if (done % 100 < dts.length) status.textContent = `${label}: ${done}/${steps} steps`;
    }
    return done;
}

async function runCase(device, kernels, name, width, height, steps) {
    const { config, initial } = scenario(name, width, height);
    const cpu = new (cpuFor(kernels.scheme))(variant(config, kernels.scheme), initial);
    const gpu = await createSolver(device, variant(config, kernels.scheme), initial, kernels);
    const volume0 = diagnostics(initial.state, config).volume;
    const started = performance.now();
    try {
        const taken = await advanceBoth(gpu, cpu, kernels, steps, 0.05, `${name} ${width}×${height}`);
        const state = await gpu.readState(), stats = diagnostics(state, config);
        const outflow = await gpu.readBoundaryOutflow();
        const absoluteBudgetResidual = stats.volume + outflow - volume0 - gpu.sourceVolume;
        // f32 budget error scales with the largest volume term (initial + sources, final, boundary flux).
        const scale = Math.max(volume0 + gpu.sourceVolume, stats.volume, Math.abs(outflow), 1e-30);
        const relativeBudgetResidual = Math.abs(absoluteBudgetResidual) / scale;
        const equilibrium = EQUILIBRIA.includes(name);
        const analytical = equilibrium || name === 'rain' || name === 'friction' || name === 'wind-uniform' || name === 'coriolis-uniform';
        const ft = 1e-3 * gpu.time, q0 = name === 'coriolis-uniform' ? [initial.state[1], initial.state[2]] : null;
        const coriolisExact = q0 ? [q0[0] * Math.cos(ft) + q0[1] * Math.sin(ft), -q0[0] * Math.sin(ft) + q0[1] * Math.cos(ft)] : null;
        const windExact = name === 'wind-uniform' ? windStress(gpu.config).map(s => s * gpu.time) : null;
        const exactFriction = name === 'friction'
            ? frictionExact(Array.from(initial.state.subarray(0, 3)), initial.manning[0], config.gravity, gpu.time) : null;
        let maxReferenceError = 0, maxAnalyticalError = analytical ? 0 : null, solidWet = 0;
        for (let k = 0; k < state.length; k++) {
            if (k % 4 === 3) continue; // lane 4 holds the GPU depth compensation, not physical state
            maxReferenceError = Math.max(maxReferenceError, Math.abs(state[k] - cpu.state[k]));
            if (initial.solid[k >> 2] && state[k] !== 0) solidWet++;
            if (equilibrium) {
                maxAnalyticalError = Math.max(maxAnalyticalError, Math.abs(state[k] - initial.state[k]));
            } else if (name === 'rain') {
                maxAnalyticalError = Math.max(maxAnalyticalError, Math.abs(state[k] - (k % 4 === 0 ? config.rain * gpu.time : 0)));
            } else if (coriolisExact && k % 4 < 3) {
                // Inertial oscillation of a uniform current: exact rotation at constant depth.
                maxAnalyticalError = Math.max(maxAnalyticalError, Math.abs(state[k] - (k % 4 === 0 ? initial.state[k] : coriolisExact[k % 4 - 1])));
            } else if (windExact && k % 4 < 3) {
                // Uniform wind on a periodic domain: q = tau t exactly, depth unchanged.
                maxAnalyticalError = Math.max(maxAnalyticalError, Math.abs(state[k] - (k % 4 === 0 ? initial.state[k] : windExact[k % 4 - 1])));
            } else if (exactFriction && k % 4 < 3) {
                maxAnalyticalError = Math.max(maxAnalyticalError, Math.abs(state[k] - exactFriction[k % 4]));
            }
        }
        const checks = {
            finiteAndNonnegative: stats.minDepth >= 0,
            massBudget: relativeBudgetResidual <= 1e-5 || Math.abs(absoluteBudgetResidual) / (width * height * config.dx * config.dy) <= 1e-8,
            cpuAgreement: maxReferenceError <= referenceTolerance(kernels.scheme, name),
            clockAgreement: Math.abs(gpu.time - cpu.time) <= 1e-4,
            // Outflow integrates thousands of f32 face fluxes; its relative difference tracks the
            // state difference (Rusanov outflow-drain: 2.6e-5 state, 2.4e-5 outflow).
            outflowAgreement: Math.abs(outflow - cpu.boundaryOutflow) <= 1e-4 * Math.max(1, Math.abs(cpu.boundaryOutflow)),
            solidCellsDry: solidWet === 0,
        };
        if (analytical) checks.analytical = maxAnalyticalError <= (name === 'rain' ? 1e-8 : 1e-5);
        if (['lake', 'lake-rounded', 'shoreline', 'lake-obstacle', 'stage-lake', 'wave-lake'].includes(name)) {
            checks.restSpeed = stats.maxSpeed <= 1e-5 * Math.sqrt(config.gravity * 2);
        }
        const result = { name, width, height, steps: taken, time: gpu.time, checks,
            passed: Object.values(checks).every(Boolean), ...stats, boundaryOutflow: outflow, sourceVolume: gpu.sourceVolume,
            absoluteBudgetResidual, relativeBudgetResidual, maxReferenceError, maxAnalyticalError,
            allocatedBytes: gpu.allocatedBytes, counters: { ...gpu.counters },
            wallMilliseconds: performance.now() - started };
        // Timings include reference execution and readback, not a benchmark.
        return { result, state };
    } finally { gpu.destroy(); }
}

async function exactCase(device, kernels, name, width) {
    const fixture = name === 'diagonal-wave' ? diagonalScenario(width) : name === 'steady-bed'
        ? steadyBedScenario(width) : exactScenario(name, width);
    const { config, initial, finalTime, sample } = fixture;
    const cpu = new (cpuFor(kernels.scheme))(variant(config, kernels.scheme), initial);
    const gpu = await createSolver(device, variant(config, kernels.scheme), initial, kernels);
    const initialVolume = diagnostics(initial.state, config).volume;
    try {
        status.textContent = `${kernels.scheme}: ${name}, ${width} cells`;
        const { dts } = await gpu.advance(finalTime, { maxDt: finalTime });
        for (const dt of dts) cpu.step(dt);
        const state = await gpu.readState(), stats = diagnostics(state, config);
        const exact = name === 'diagonal-wave' ? diagonalAverages(width, gpu.time) : cellAverages(width, config.dx, sample, gpu.time);
        const errors = depthErrors(state, exact);
        let referenceError = 0;
        for (let i = 0; i < state.length; i++) if (i % 4 !== 3) referenceError = Math.max(referenceError, Math.abs(state[i] - cpu.state[i]));
        const massResidual = Math.abs(stats.volume - initialVolume) / initialVolume;
        return { name, scheme: kernels.scheme, width, steps: dts.length, time: gpu.time, depthErrors_m: errors,
            massResidual, referenceError, counters: { ...gpu.counters },
            passed: referenceError <= 2e-4 && massResidual <= 1e-5 && Math.abs(gpu.time - cpu.time) <= 1e-4
                && Math.abs(gpu.time - finalTime) <= 1e-5 * finalTime };
    } finally { gpu.destroy(); }
}

// Fused and multi-pass MUSCL kernels implement the same discretization; compare
// them step by step with identical dt caps, across workgroup seams and forcing.
async function equivalence(device, fusedKernels, referenceKernels, name, width, height, steps) {
    const { config, initial } = scenario(name, width, height);
    const a = await createSolver(device, variant(config, 'muscl-fused'), initial, fusedKernels);
    const b = await createSolver(device, variant(config, referenceKernels.scheme), initial, referenceKernels);
    try {
        let maxDtDifference = 0;
        for (let i = 0; i < steps; i++) {
            const dtA = await a.step(0.05), dtB = await b.step(0.05);
            maxDtDifference = Math.max(maxDtDifference, Math.abs(dtA - dtB));
        }
        const [sa, sb] = [await a.readState(), await b.readState()];
        let maxStateDifference = 0;
        for (let k = 0; k < sa.length; k++) maxStateDifference = Math.max(maxStateDifference, Math.abs(sa[k] - sb[k]));
        const [oa, ob] = [await a.readBoundaryOutflow(), await b.readBoundaryOutflow()];
        const countersMatch = a.counters.hlleFallbacks === b.counters.hlleFallbacks
            && a.counters.reconstructionFallbacks === b.counters.reconstructionFallbacks;
        return { name, width, height, steps, maxStateDifference, maxDtDifference, outflowDifference: Math.abs(oa - ob),
            fusedCounters: { ...a.counters }, referenceCounters: { ...b.counters }, countersMatch,
            passed: maxStateDifference <= 1e-5 && maxDtDifference <= 1e-9 && countersMatch };
    } finally { a.destroy(); b.destroy(); }
}

const show = extra => { output.textContent = JSON.stringify({ ...report, ...extra }, null, 2); };

run.addEventListener('click', async () => {
    run.disabled = true; download.disabled = true; method.disabled = true;
    const scheme = method.value;
    report = { timestamp: new Date().toISOString(), kind: `${scheme}-correctness`,
        userAgent: navigator.userAgent, lowFroude, cases: [], errors: [],
        thresholds: { relativeMass: 1e-5, meanDepthMassAbsolute: 1e-8,
            maxReferenceStateError: 2e-4, relativeOutflowAgreement: 1e-4, maxEquilibriumError: 1e-5, maxRainError: 1e-8, fusedEquivalence: 1e-5 },
        note: 'Wall timings include CPU reference and readback; not performance measurements.' };
    let device;
    try {
        if (!navigator.gpu) throw new Error('WebGPU unavailable');
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
            device: adapter.info?.device, description: adapter.info?.description };
        device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, scheme);
        report.shaderSha256 = kernels.sha256;
        report.compilerMessages = kernels.messages;
        for (const name of ['constant', 'lake', 'lake-rounded', 'shoreline', 'rain', 'dam', 'dry-dam', 'pulse',
            'friction', 'sources', 'obstacle', 'lake-obstacle', 'open-uniform', 'open-dam', 'outflow-drain',
            'stage-lake', 'hydrograph-channel', 'wave-lake', 'wave-channel', 'wind-uniform', 'wind-basin', 'coriolis-uniform', 'coriolis-basin']) {
            const { result } = await runCase(device, kernels, name, 17, 33, 1000);
            report.cases.push(result); show();
        }
        for (const [width, height] of [[1, 1], [7, 19], [16, 16], [17, 33], [65, 65], [31, 47], [48, 32]]) {
            for (const name of ['pulse', 'obstacle', 'open-dam', 'hydrograph-channel', 'wave-channel', 'wind-basin', 'coriolis-basin']) {
                const { result } = await runCase(device, kernels, name, width, height, 100);
                report.cases.push(result); show();
            }
        }
        let first;
        const differences = [];
        for (let i = 0; i < 5; i++) {
            const { result, state } = await runCase(device, kernels, 'pulse', 17, 33, 100);
            report.cases.push(result);
            if (!first) first = state;
            let maximum = 0;
            for (let k = 0; k < state.length; k++) maximum = Math.max(maximum, Math.abs(first[k] - state[k]));
            differences.push(maximum);
        }
        report.repeatability = { runs: 5, maxStateDifferences: differences, passed: differences.every(v => v === 0) };
        report.passed = report.cases.every(c => c.passed) && report.repeatability.passed;
        // Run-time changes: setHydrographs (a raised inflow) and setParameters (wind, Coriolis) applied
        // to the GPU solver and the f64 reference at the same step must keep them together.
        {
            report.runtime = [];
            for (const name of ['hydrograph-channel', 'wind-basin']) {
                const { config, initial } = scenario(name, 37, 29);
                const cpu = new (cpuFor(kernels.scheme))(variant(config, kernels.scheme), initial);
                const gpu = await createSolver(device, variant(config, kernels.scheme), initial, kernels);
                try {
                    await advanceBoth(gpu, cpu, kernels, 150, 0.05, `${name} runtime`);
                    if (name === 'hydrograph-channel') {
                        const changed = [{ ...config.hydrographs[0], series: [[gpu.time, 2 * Math.min(1, gpu.time / 20)], [gpu.time + 4, 3.5]] }, config.hydrographs[1]];
                        gpu.setHydrographs(changed); cpu.setHydrographs(changed);
                    } else {
                        gpu.setParameters({ wind: [-12, 9], coriolis: 5e-4 }); cpu.setParameters({ wind: [-12, 9], coriolis: 5e-4 });
                    }
                    await advanceBoth(gpu, cpu, kernels, 300, 0.05, `${name} runtime`);
                    const state = await gpu.readState();
                    let maxReferenceError = 0;
                    for (let k = 0; k < state.length; k++) if (k % 4 !== 3) maxReferenceError = Math.max(maxReferenceError, Math.abs(state[k] - cpu.state[k]));
                    report.runtime.push({ name, maxReferenceError, passed: maxReferenceError <= referenceTolerance() });
                } finally { gpu.destroy(); }
            }
            report.passed = report.passed && report.runtime.every(r => r.passed);
        }
        // setBed (interactive terrain): depths are unchanged by the edit, and the flow that the raised
        // bed then drives keeps depths nonnegative and closes the water budget.
        {
            const { config, initial } = scenario('lake', 17, 33);
            const gpu = await createSolver(device, variant(config, kernels.scheme), initial, kernels);
            try {
                const before = await gpu.readState(), raised = [], elevations = [];
                for (let y = 10; y < 16; y++) for (let x = 5; x < 11; x++) { raised.push(y * 17 + x); elevations.push(initial.bed[y * 17 + x] + 0.25); }
                gpu.setBed(raised, elevations);
                const after = await gpu.readState();
                let depthChange = 0;
                for (let k = 0; k < after.length; k += 4) depthChange = Math.max(depthChange, Math.abs(after[k] - before[k]));
                const volume0 = diagnostics(after, config).volume;
                await advanceBoth(gpu, { step() {} }, kernels, 200, 0.05, 'setBed');
                const state = await gpu.readState(), stats = diagnostics(state, config), outflow = await gpu.readBoundaryOutflow();
                const residual = Math.abs(stats.volume + outflow - volume0) / volume0;
                report.setBed = { depthChange, residual, minDepth: stats.minDepth, passed: depthChange === 0 && residual < 1e-6 && stats.minDepth >= 0 };
                report.passed = report.passed && report.setBed.passed;
            } finally { gpu.destroy(); }
        }
        // Sparse sources (setSources) and state replacement (setState, and the same edit written directly
        // into currentState as an external GPU pass would) applied to the GPU solver and the f64
        // reference at the same step must keep them together. Rates are multiples of 2^-14 m/s, exact in
        // f32, so both budgets see the same inflow.
        {
            const { config, initial } = scenario('pulse', 37, 29), n = 37 * 29;
            const rows = [];
            for (const path of ['setSources', 'setState', 'external edit']) {
                const cpu = new (cpuFor(kernels.scheme))(variant(config, kernels.scheme), initial);
                const gpu = await createSolver(device, variant(config, kernels.scheme), initial, kernels);
                try {
                    let previous = [], dischargeError = 0, editVolumes = null;
                    for (let f = 0; f < 8; f++) {
                        const now = [3 + 4 * f, 37 * 14 + 2 + 3 * f, 37 * 20 + 30 - f].map((i, k) => [i, (k + 1 + f) * 2 ** -14]);
                        const indices = [...previous.map(([i]) => i), ...now.map(([i]) => i)];
                        const rates = [...previous.map(() => 0), ...now.map(([, r]) => r)];
                        gpu.setSources(indices, rates); cpu.setSources(indices, rates);
                        const full = sourceDischarge(gpu.config, gpu.forcing);
                        dischargeError = Math.max(dischargeError, Math.abs(gpu.discharge - full) / full);
                        await advanceBoth(gpu, cpu, kernels, 25, 0.05, `${path} frame ${f}`);
                        previous = now;
                        if (f === 3 && path !== 'setSources') {
                            // Raise the western third by 0.05 m at constant velocity and dry one row.
                            const edited = new Float32Array(cpu.state);
                            for (let i = 0; i < n; i++) {
                                if (i % 37 > 12) continue;
                                const h = edited[4 * i], scale = h > 0 ? (h + 0.05) / h : 0;
                                edited[4 * i] = h + 0.05; edited[4 * i + 1] *= scale; edited[4 * i + 2] *= scale;
                            }
                            edited.fill(0, 4 * 37 * 28, 4 * n); edited.forEach((_, k) => { if (k % 4 === 3) edited[k] = 0; });
                            const cpuBefore = cpu.stateEditVolume;
                            cpu.setState(Float64Array.from(edited));
                            if (path === 'setState') await gpu.setState(edited);
                            else device.queue.writeBuffer(gpu.currentState, 0, edited); // external edit (outside the budget)
                            editVolumes = { gpu: gpu.stateEditVolume, cpu: cpu.stateEditVolume - cpuBefore };
                        }
                    }
                    const state = await gpu.readState();
                    let maxReferenceError = 0;
                    for (let k = 0; k < state.length; k++) if (k % 4 !== 3) maxReferenceError = Math.max(maxReferenceError, Math.abs(state[k] - cpu.state[k]));
                    const sourceVolumeError = Math.abs(gpu.sourceVolume - cpu.sourceVolume) / cpu.sourceVolume;
                    // setState reads the GPU state it replaces, so its edit volume differs from the
                    // reference's by at most the state difference at that time over the edited cells.
                    const editOk = path !== 'setState' || Math.abs(editVolumes.gpu - editVolumes.cpu) <= referenceTolerance() * n * config.dx * config.dy;
                    rows.push({ path, maxReferenceError, dischargeError, sourceVolumeError, editVolumes,
                        passed: maxReferenceError <= referenceTolerance() && dischargeError < 1e-12 && sourceVolumeError < 1e-6 && editOk });
                } finally { gpu.destroy(); }
            }
            report.stateAndSources = rows;
            report.passed = report.passed && rows.every(r => r.passed);
        }
        if (scheme !== 'rusanov') {
            const firstOrder = await compileKernels(device);
            report.baselineShaderSha256 = firstOrder.sha256;
            const refinements = [];
            for (const width of [32, 64, 128, 256]) {
                refinements.push(await exactCase(device, kernels, 'simple-wave', width));
                show({ refinements });
            }
            const orders = refinements.slice(1).map((r, i) => Math.log2(refinements[i].depthErrors_m.l1 / r.depthErrors_m.l1));
            const control = await exactCase(device, firstOrder, 'simple-wave', 256);
            report.convergence = { refinements, orders, control,
                passed: refinements.every(r => r.passed) && control.passed && orders.at(-1) >= 1.8
                    && refinements.at(-1).depthErrors_m.l1 < 0.25 * control.depthErrors_m.l1 };
            report.damAccuracy = [];
            for (const name of ['wet-dam', 'dry-dam']) {
                const baseline = await exactCase(device, firstOrder, name, 256);
                const muscl = await exactCase(device, kernels, name, 256);
                const ratio = muscl.depthErrors_m.l1 / baseline.depthErrors_m.l1;
                report.damAccuracy.push({ name, baseline, muscl, ratio, passed: baseline.passed && muscl.passed && ratio < 0.8 });
            }
            report.additionalConvergence = [];
            for (const [name, widths] of [['diagonal-wave', [16, 32, 64]], ['steady-bed', [16, 32, 64]]]) {
                const rows = [];
                for (const width of widths) rows.push(await exactCase(device, kernels, name, width));
                const orders = rows.slice(1).map((r, i) => Math.log2(rows[i].depthErrors_m.l1 / r.depthErrors_m.l1));
                report.additionalConvergence.push({ name, rows, orders,
                    passed: rows.every(r => r.passed) && orders.at(-1) >= 1.8 });
            }
            report.passed = report.passed && report.convergence.passed && report.damAccuracy.every(r => r.passed)
                && report.additionalConvergence.every(r => r.passed);
        }
        if (scheme === 'muscl-fused') {
            const multiPass = await compileKernels(device, 'muscl-hancock');
            report.multiPassShaderSha256 = multiPass.sha256;
            report.equivalence = [];
            for (const [name, width, height] of [['pulse', 65, 65], ['obstacle', 37, 29], ['open-dam', 50, 40],
                ['sources', 33, 17], ['friction', 16, 48], ['dry-dam', 129, 3], ['pulse', 1, 1],
                ['hydrograph-channel', 50, 40], ['stage-lake', 37, 29], ['wave-channel', 50, 40], ['wave-lake', 37, 29], ['wind-basin', 50, 40], ['coriolis-basin', 50, 40]]) {
                report.equivalence.push(await equivalence(device, kernels, multiPass, name, width, height, 200));
                show();
            }
            report.passed = report.passed && report.equivalence.every(r => r.passed);
            // Batch invariance: the same 300 steps in batches of 1, 16 and 256 must be bitwise equal
            // (GPU-resident timestep and the two-float hydrograph clock make steps independent of batching).
            report.batchInvariance = [];
            for (const [name, width, height] of [['hydrograph-channel', 50, 40], ['obstacle', 37, 29], ['sources', 33, 17], ['wave-channel', 50, 40], ['wind-basin', 50, 40], ['coriolis-basin', 50, 40]]) {
                const states = [];
                for (const batch of [1, 16, 256]) {
                    const { config, initial } = scenario(name, width, height);
                    const solver = await createSolver(device, variant(config, kernels.scheme), initial, kernels);
                    try {
                        await solver.advance(1e6, { maxDt: 0.05, batch, maxSteps: 300 });
                        states.push({ state: await solver.readState(), outflow: await solver.readBoundaryOutflow(), time: solver.time });
                    } finally { solver.destroy(); }
                }
                const same = states.every(x => x.time === states[0].time && x.outflow === states[0].outflow
                    && x.state.every((v, k) => v === states[0].state[k]));
                report.batchInvariance.push({ name, width, height, batches: [1, 16, 256], steps: 300, passed: same });
                show();
            }
            report.passed = report.passed && report.batchInvariance.every(r => r.passed);
        }
        report.passed = report.passed && report.errors.length === 0;
    } catch (error) {
        report.errors.push(error.message);
        report.passed = false;
    } finally {
        device?.destroy();
        window.__validation = report;
        show();
        status.textContent = report.passed ? `PASS — ${scheme} validation suite completed.` : 'FAIL — inspect report.';
        run.disabled = false; download.disabled = false; method.disabled = false;
    }
});

download.addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `${method.value}-validation.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
