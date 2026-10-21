import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { MusclReferenceSolver } from '../../src/muscl-reference.mjs';
import { scenario } from '../../src/scenarios.mjs';
import { thackerScenario } from '../../src/benchmarks.mjs';
import { diagnostics } from '../../src/model.mjs';

// Elevation-offset sweep: the same problem lifted by Z metres (bed, and stage hydrographs), run
// on the fused f32 kernel with the automatic datum shift and without it (datum 0: raw f32
// elevations), and compared with the f64 MUSCL reference at the same offset.
const status = document.querySelector('#status'), output = document.querySelector('#output');
const OFFSETS = [0, 10, 100, 1000, 3000, 8000];
const CASES = {
    'lake-rounded': () => scenario('lake-rounded', 33, 33),   // exact discrete lake; must stay at rest
    'stage-lake': () => scenario('stage-lake', 33, 33),       // lake held by stage boundaries
    pulse: () => scenario('pulse', 33, 33),                   // smooth wet dynamics
    sources: () => scenario('sources', 33, 33),               // thin films from a dry start
    thacker: () => thackerScenario(64),                        // moving shoreline
};
const STEPS = 400;

function lift({ config, initial }, offset) {
    const bed = Float64Array.from(initial.bed, z => z + offset); // f64: the physical problem is unchanged
    const hydrographs = (config.hydrographs ?? []).map(h => h.kind === 'stage'
        ? { ...h, series: h.series.map(([t, v]) => [t, v + offset]) } : h);
    return { config: { ...config, hydrographs }, initial: { ...initial, bed } };
}

async function runCase(device, kernels, name, offset, datum) {
    const base = lift(CASES[name](), offset), config = { ...base.config, datum };
    const gpu = await createSolver(device, config, base.initial, kernels);
    const cpu = new MusclReferenceSolver({ ...base.config, datum: 'auto' }, base.initial);
    try {
        let done = 0;
        while (done < STEPS) {
            const { dts } = await gpu.advance(1e6, { maxDt: 0.05, batch: 64, maxSteps: STEPS - done });
            for (const dt of dts) cpu.step(dt);
            done += dts.length;
        }
        const state = await gpu.readState();
        let maxDepth = 0, maxMomentum = 0, rest = 0;
        for (let i = 0; i < state.length; i += 4) {
            maxDepth = Math.max(maxDepth, Math.abs(state[i] - cpu.state[i]));
            maxMomentum = Math.max(maxMomentum, Math.abs(state[i + 1] - cpu.state[i + 1]), Math.abs(state[i + 2] - cpu.state[i + 2]));
            rest = Math.max(rest, Math.abs(state[i] - base.initial.state[i]), Math.abs(state[i + 1]), Math.abs(state[i + 2]));
        }
        const v0 = diagnostics(base.initial.state, base.config).volume;
        const residual = diagnostics(state, base.config).volume + await gpu.readBoundaryOutflow() - v0 - gpu.sourceVolume;
        return { name, offset, datum, steps: done, maxDepthErrorVsF64: maxDepth, maxMomentumErrorVsF64: maxMomentum,
            maxChangeFromInitial: rest, relativeBudgetResidual: Math.abs(residual) / Math.max(v0 + gpu.sourceVolume, 1e-30),
            retries: gpu.counters.retries };
    } catch (error) {
        return { name, offset, datum, fatal: error.message };
    } finally { gpu.destroy(); }
}

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'precision', userAgent: navigator.userAgent,
        steps: STEPS, offsets: OFFSETS, rows: [], errors: [] };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = kernels.sha256;
        for (const name of Object.keys(CASES)) for (const offset of OFFSETS) for (const datum of ['auto', 0]) {
            status.textContent = `${name}, offset ${offset} m, datum ${datum}`;
            report.rows.push(await runCase(device, kernels, name, offset, datum));
            output.textContent = JSON.stringify(report.rows.at(-1));
        }
        device.destroy();
    } catch (error) { report.errors.push(error.message); }
    window.__precisionReport = report;
    output.textContent = JSON.stringify(report, null, 1);
    status.textContent = 'Done.';
});
