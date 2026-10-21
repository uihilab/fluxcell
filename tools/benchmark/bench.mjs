import { createSolver, compileKernels } from '../../src/webgpu.mjs';
import { scenario } from '../../src/scenarios.mjs';

const runButton = document.querySelector('#run'), download = document.querySelector('#download');
const status = document.querySelector('#status'), output = document.querySelector('#output');
const tbody = document.querySelector('#table tbody'), largest = document.querySelector('#largest');
let report;

const VARIANTS = [
    { label: 'multi-pass', scheme: 'muscl-hancock', steps: 16, batch: 1 },
    { label: 'fused, 1 step/submit', scheme: 'muscl-fused', steps: 16, batch: 1 },
    { label: 'fused, batched', scheme: 'muscl-fused', steps: 64, batch: 64 },
];
// Fully wet variable-bed pulse and a dam break onto a dry bed (moving shoreline).
const WORKLOADS = ['pulse', 'dry-dam'];

const quantile = (values, q) => { const s = [...values].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

async function measure(device, kernels, variant, workload, size) {
    const { config, initial } = scenario(workload, size, size);
    const solver = await createSolver(device, config, initial, kernels);
    try {
        const options = { maxDt: 0.05, maxSteps: variant.steps, batch: variant.batch };
        await solver.advance(1e6, { ...options, maxSteps: 20 });
        const perStep = [];
        let steps = 0;
        const started = performance.now();
        for (let i = 0; i < 30; i++) {
            const t0 = performance.now();
            const { steps: taken } = await solver.advance(1e6, options);
            perStep.push((performance.now() - t0) / taken);
            steps += taken;
        }
        const seconds = (performance.now() - started) / 1000;
        return { workload, size, variant: variant.label, scheme: variant.scheme, stepsPerBatch: variant.steps,
            batches: 30, steps, medianMsPerStep: quantile(perStep, 0.5), p95MsPerStep: quantile(perStep, 0.95),
            cellStepsPerSecond: size * size * steps / seconds, simulatedSecondsPerWallSecond: solver.time / seconds,
            allocatedBytes: solver.allocatedBytes, counters: { ...solver.counters } };
    } finally { solver.destroy(); }
}

function addRow(row, baseline) {
    const tr = document.createElement('tr');
    const speedup = baseline ? (baseline.medianMsPerStep / row.medianMsPerStep).toFixed(2) + '×' : '—';
    for (const value of [row.workload, `${row.size}²`, row.variant, row.stepsPerBatch, row.medianMsPerStep.toFixed(3),
        row.p95MsPerStep.toFixed(3), (row.cellStepsPerSecond / 1e6).toFixed(1), speedup, (row.allocatedBytes / 2 ** 20).toFixed(1)]) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    tbody.append(tr);
}

runButton.addEventListener('click', async () => {
    runButton.disabled = true; download.disabled = true; tbody.replaceChildren();
    report = { timestamp: new Date().toISOString(), kind: 'throughput', userAgent: navigator.userAgent, rows: [], errors: [],
        method: 'Warm-up 20 steps; 30 timed batches; wall time to queue completion including control readback; no rendering.' };
    let device;
    try {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter');
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
            device: adapter.info?.device, description: adapter.info?.description };
        device = await adapter.requestDevice({ requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = {};
        for (const scheme of ['muscl-hancock', 'muscl-fused']) kernels[scheme] = await compileKernels(device, scheme);
        report.shaderSha256 = Object.fromEntries(Object.entries(kernels).map(([k, v]) => [k, v.sha256]));
        const sizes = [128, 256, 512, 1024, 2048].filter(s => s <= Number(largest.value));
        for (const workload of WORKLOADS) for (const size of sizes) {
            let baseline;
            for (const variant of VARIANTS) {
                status.textContent = `${workload} ${size}²: ${variant.label}`;
                try {
                    const row = await measure(device, kernels[variant.scheme], variant, workload, size);
                    baseline ??= row;
                    row.speedupVsMultiPass = baseline.medianMsPerStep / row.medianMsPerStep;
                    report.rows.push(row); addRow(row, baseline);
                } catch (error) { report.errors.push(`${workload} ${size} ${variant.label}: ${error.message}`); }
                output.textContent = JSON.stringify(report, null, 2);
            }
        }
        status.textContent = report.errors.length ? 'Completed with errors — inspect report.' : 'Benchmark complete.';
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
    link.href = url; link.download = 'muscl-throughput.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
