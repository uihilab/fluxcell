import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { scenario } from '../../src/scenarios.mjs';
import { thackerScenario, thackerAverages } from '../../src/benchmarks.mjs';

// GPU kernel time from timestamp queries, end-to-end throughput by batch size, and cost to reach
// a given accuracy (Thacker paraboloid) for fused MUSCL and first-order Rusanov.
const status = document.querySelector('#status'), output = document.querySelector('#output');
const median = v => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

let device, querySet, resolveBuffer, readBuffer;
const TS = () => ({ querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
async function gpuNanoseconds() {
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const [a, b] = new BigInt64Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    return Number(b - a);
}

// Median GPU ms per step: fused over 256-step batches, multi-pass per step.
async function kernelTime(kernels, workload, size) {
    const { config, initial } = scenario(workload, size, size);
    const solver = await createSolver(device, config, initial, kernels);
    try {
        const fused = kernels.scheme === 'muscl-fused', samples = [];
        if (fused) {
            await solver.advance(1e6, { maxDt: 0.05, batch: 64, maxSteps: 64 });
            for (let r = 0; r < 5; r++) {
                const { steps } = await solver.advance(1e6, { maxDt: 0.05, batch: 256, maxSteps: 256, timestamps: TS() });
                samples.push(await gpuNanoseconds() / 1e6 / steps);
            }
        } else {
            for (let k = 0; k < 10; k++) await solver.step(0.05);
            for (let k = 0; k < 20; k++) { await solver.step(0.05, { timestamps: TS() }); samples.push(await gpuNanoseconds() / 1e6); }
        }
        const ms = median(samples);
        return { scheme: kernels.scheme, workload, size, gpuMsPerStep: ms, gpuCellStepsPerSecond: size * size / (ms / 1000),
            bytesPerCell: solver.allocatedBytes / (size * size) };
    } finally { solver.destroy(); }
}

async function batchSweep(kernels, size) {
    const rows = [];
    for (const batch of [1, 2, 4, 8, 16, 32, 64, 128, 256]) {
        const { config, initial } = scenario('pulse', size, size);
        const solver = await createSolver(device, config, initial, kernels);
        try {
            await solver.advance(1e6, { maxDt: 0.05, batch, maxSteps: 64 });
            const t0 = performance.now(), { steps } = await solver.advance(1e6, { maxDt: 0.05, batch, maxSteps: 512 });
            const seconds = (performance.now() - t0) / 1000;
            rows.push({ batch, size, steps, wallMsPerStep: 1000 * seconds / steps, cellStepsPerSecond: size * size * steps / seconds });
        } finally { solver.destroy(); }
        status.textContent = `batch sweep ${size}²: batch ${batch}`;
    }
    return rows;
}

// Thacker to one period; GPU time summed over batches (fused) or steps (multi-pass).
async function accuracyCost(kernels, n) {
    const { config, initial, period } = thackerScenario(n);
    const solver = await createSolver(device, config, initial, kernels);
    try {
        let gpuMs = 0, steps = 0;
        const t0 = performance.now();
        if (kernels.scheme === 'muscl-fused') {
            await solver.advance(Math.min(period, 1e-3), { batch: 1 });
            steps = 1;
            while (period - solver.time > 1e-9 * period) {
                const r = await solver.advance(period - solver.time, { batch: 256, maxSteps: 256, timestamps: TS() });
                gpuMs += await gpuNanoseconds() / 1e6; steps += r.steps;
            }
        } else {
            while (period - solver.time > 1e-9 * period) {
                await solver.step(period - solver.time, { timestamps: TS() });
                gpuMs += await gpuNanoseconds() / 1e6; steps++;
            }
        }
        const wallSeconds = (performance.now() - t0) / 1000;
        const state = await solver.readState(), exact = thackerAverages(n, solver.time).state;
        let l1 = 0;
        for (let k = 0; k < exact.length; k += 4) l1 += Math.abs(state[k] - exact[k]);
        return { scheme: kernels.scheme, n, steps, cellSteps: n * n * steps, l1: l1 / (n * n), gpuMs, wallSeconds };
    } finally { solver.destroy(); }
}

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'performance', userAgent: navigator.userAgent, errors: [],
        method: 'GPU time from timestamp queries (fused: 256-step batches incl. one wave-speed pass; multi-pass: per step, all passes). '
            + 'Batch sweep: wall time incl. submission and the control readback per batch. No rendering.',
        kernel: [], batch: [], accuracy: [] };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'], requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
        resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        readBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const kernels = {};
        for (const scheme of ['muscl-fused', 'muscl-hancock', 'rusanov']) kernels[scheme] = await compileKernels(device, scheme);
        report.shaderSha256 = Object.fromEntries(Object.entries(kernels).map(([k, v]) => [k, v.sha256]));
        for (const workload of ['pulse', 'dry-dam']) for (const size of [128, 256, 512, 1024, 2048]) {
            for (const scheme of ['muscl-fused', 'muscl-hancock', 'rusanov']) {
                if (scheme !== 'muscl-fused' && size > 1024) continue; // one-dimensional dispatch limit
                status.textContent = `kernel time: ${scheme} ${workload} ${size}²`;
                try { report.kernel.push(await kernelTime(kernels[scheme], workload, size)); }
                catch (error) { report.errors.push(`${scheme} ${workload} ${size}: ${error.message}`); }
                output.textContent = JSON.stringify(report.kernel.at(-1));
            }
        }
        for (const size of [256, 1024]) report.batch.push(...await batchSweep(kernels['muscl-fused'], size));
        for (const [scheme, sizes] of [['muscl-fused', [50, 100, 200, 400]], ['rusanov', [50, 100, 200, 400, 800]]]) {
            for (const n of sizes) {
                status.textContent = `accuracy cost: ${scheme} ${n}²`;
                report.accuracy.push(await accuracyCost(kernels[scheme], n));
                output.textContent = JSON.stringify(report.accuracy.at(-1));
            }
        }
        device.destroy();
    } catch (error) { report.errors.push(error.message); }
    window.__perfReport = report;
    output.textContent = JSON.stringify(report, null, 1);
    status.textContent = report.errors.length ? 'Completed with errors.' : 'Done.';
});
