import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { shearLayerCase, wakeCase, vortexExitCase, turbulenceCase, modeAmplitude, mixedFraction, energySpectrum, energyEnstrophy,
    vorticity, MICHALKE } from '../../src/transport.mjs';
// ?lowFroude runs every case with the low-Froude correction; ?smagorinsky=C and ?bedTurbulence=a
// switch the turbulence closure on (model.mjs).
const query = new URLSearchParams(location.search);
const OPTIONS = { ...(query.has('lowFroude') ? { lowFroude: true } : {}),
    ...(query.has('smagorinsky') ? { smagorinsky: Number(query.get('smagorinsky')) } : {}),
    ...(query.has('bedTurbulence') ? { bedTurbulence: Number(query.get('bedTurbulence')) } : {}) };
const variant = c => ({ ...c, ...OPTIONS });


// Shallow-flow dynamics on the fused kernel against known physics: Kelvin-Helmholtz growth rate
// (Michalke, 1964), the Strouhal number of a vortex street behind an island, and decaying
// two-dimensional turbulence (energy, enstrophy, spectra, mixing). Fields at selected times are
// stored for figures in data/figures/ (not versioned; this page regenerates them).
const status = document.querySelector('#status'), output = document.querySelector('#output');
const say = text => { status.textContent = text; };
const b64 = array => { let s = ''; const bytes = new Uint8Array(new Float32Array(array).buffer); for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };
const fit = (t, y) => { // least-squares slope
    const n = t.length, mt = t.reduce((a, b) => a + b) / n, my = y.reduce((a, b) => a + b) / n;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (t[i] - mt) * (y[i] - my); den += (t[i] - mt) ** 2; }
    return num / den;
};
const sumOf = a => { let s = 0; for (const v of a) s += v; return s; };

// Reads [h, hu, hv, comp] of a few cells from the solver's current state buffer.
async function probe(device, solver, cells) {
    const buffer = device.createBuffer({ size: 16 * cells.length, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    cells.forEach((c, k) => encoder.copyBufferToBuffer(solver.currentState, 16 * c, buffer, 16 * k, 16));
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.destroy();
    return values;
}

function fields(state, amount, width, height, dx, periodic) {
    const w = vorticity(state, width, height, dx, dx, periodic);
    const c = Array.from(amount, (a, i) => (state[4 * i] > 0 ? a / state[4 * i] : 0));
    return { vorticity: b64(w), concentration: b64(c) };
}

async function kelvinHelmholtz(device, kernels, sizes, store, depth = 10) {
    const rows = [];
    for (const n of sizes) {
        const c = shearLayerCase(n, { depth }), gpu = await createSolver(device, variant(c.config), c.initial, kernels);
        const series = [];
        try {
            for (let t = 0; t <= 80; t += 1) {
                say(`Kelvin-Helmholtz n = ${n}, depth ${depth} m, t = ${t} s`);
                await gpu.advance(t - gpu.time > 0 ? t - gpu.time : 1e-9, { batch: 256 });
                const state = await gpu.readState(), amount = await gpu.readTracer();
                series.push({ t: gpu.time, amplitude: modeAmplitude(state, n, c.modes), mixed: mixedFraction(state, amount) });
                if (store && n === sizes.at(-1) && [20, 30, 40, 50, 60, 80].includes(t)) {
                    store.kh ??= { width: n, height: n, dx: c.config.dx, times: [] };
                    store.kh.times.push({ t: gpu.time, ...fields(state, amount, n, n, c.config.dx, false) });
                }
            }
        } finally { gpu.destroy(); }
        // Linear window: amplitude between 5x and 100x its initial value (well below saturation).
        const a0 = series[0].amplitude, window = series.filter(s => s.amplitude > 5 * a0 && s.amplitude < 100 * a0);
        const sigma = window.length > 2 ? fit(window.map(s => s.t), window.map(s => Math.log(s.amplitude))) : NaN;
        rows.push({ n, depth, froude: c.U / Math.sqrt(9.81 * depth), growthRate: sigma, normalized: sigma * c.delta / c.U, michalke: MICHALKE.growthRate,
            window: window.length ? [window[0].t, window.at(-1).t] : null, series });
    }
    return rows;
}

async function wake(device, kernels, spacings, store) {
    const rows = [];
    for (const dx of spacings) {
        const c = wakeCase(dx), gpu = await createSolver(device, variant(c.config), c.initial, kernels);
        const times = [], v = [];
        try {
            for (let t = 0.5; t <= 1500; t += 0.5) {
                await gpu.advance(t - gpu.time, { batch: 256 });
                const s = await probe(device, gpu, [c.probe]);
                times.push(gpu.time); v.push(s[0] > 0 ? s[2] / s[0] : 0);
                if (t % 50 === 0) say(`wake dx = ${dx} m, t = ${t} s`);
            }
            if (store && dx === spacings.at(-1)) {
                const state = await gpu.readState(), amount = await gpu.readTracer();
                store.wake = { width: c.config.width, height: c.config.height, dx, t: gpu.time, ...fields(state, amount, c.config.width, c.config.height, dx, false) };
            }
        } finally { gpu.destroy(); }
        // Shedding frequency from upward zero crossings of the transverse velocity after 700 s.
        const crossings = [];
        for (let k = 1; k < v.length; k++) if (times[k] > 700 && v[k - 1] < 0 && v[k] >= 0) {
            crossings.push(times[k - 1] + (times[k] - times[k - 1]) * (-v[k - 1]) / (v[k] - v[k - 1]));
        }
        const period = crossings.length > 1 ? (crossings.at(-1) - crossings[0]) / (crossings.length - 1) : NaN;
        rows.push({ dx, cells: c.config.width * c.config.height, period, strouhal: c.diameter / (period * c.velocity),
            cycles: crossings.length - 1, amplitude: Math.max(...v.slice(-600).map(Math.abs)), probe: { t: times, v } });
    }
    return rows;
}

// A vortex leaving through a stage boundary (transport.mjs: vortexExitCase): the swirl energy must
// leave with it, the speed must stay bounded by its initial maximum, and no step may be retried.
async function vortexExit(device, kernels) {
    const rows = [];
    for (const dx of [0.5, 0.25, 0.125]) for (const [sign, yc] of [[-1, 8], [1, 8], [-1, 20]]) {
        const c = vortexExitCase(dx, { sign, yc }), gpu = await createSolver(device, variant(c.config), c.initial, kernels);
        const e0 = c.swirlEnergy(c.initial.state), series = [];
        let peak = 0;
        try {
            for (let t = 0; t <= 200; t += 5) {
                if (t > 0) await gpu.advance(t - gpu.time, { batch: 256 });
                const s = await gpu.readState();
                let m = 0;
                for (let i = 0; i < s.length; i += 4) if (s[i] > 0) m = Math.max(m, Math.hypot(s[i + 1], s[i + 2]) / s[i]);
                peak = Math.max(peak, m);
                series.push({ t, maxSpeed: m, energy: c.swirlEnergy(s) / e0 });
                say(`vortex exit dx = ${dx} m, t = ${t} s`);
            }
        } finally { gpu.destroy(); }
        const final = series.at(-1).energy, initialSpeed = series[0].maxSpeed;
        rows.push({ dx, sign, yc, initialSpeed, peakSpeed: peak, finalEnergyFraction: final, retries: gpu.counters.retries,
            passed: peak <= 1.05 * initialSpeed && final < 0.05 && gpu.counters.retries === 0, series });
    }
    return rows;
}

async function turbulence(device, kernels, sizes, store) {
    const rows = [];
    for (const n of sizes) {
        const c = turbulenceCase(n), gpu = await createSolver(device, variant(c.config), c.initial, kernels);
        const series = [], spectra = [];
        const mass0 = sumOf(Array.from({ length: n * n }, (_, i) => c.initial.state[4 * i]));
        const amount0 = sumOf(Array.from({ length: n * n }, (_, i) => c.initial.state[4 * i] * c.initial.concentration[i]));
        let massDrift = 0, amountDrift = 0;
        try {
            for (let t = 0; t <= 300; t += 5) {
                say(`turbulence n = ${n}, t = ${t} s`);
                if (t > 0) await gpu.advance(t - gpu.time, { batch: 256 });
                const state = await gpu.readState(), amount = await gpu.readTracer();
                const { energy, enstrophy } = energyEnstrophy(state, n, c.config.dx);
                series.push({ t: gpu.time, energy, enstrophy, mixed: mixedFraction(state, amount) });
                let mass = 0; for (let i = 0; i < n * n; i++) mass += state[4 * i];
                massDrift = Math.max(massDrift, Math.abs(mass - mass0) / mass0);
                amountDrift = Math.max(amountDrift, Math.abs(sumOf(amount) - amount0) / amount0);
                if ([0, 50, 150, 300].includes(t)) {
                    spectra.push({ t: gpu.time, E: Array.from(energySpectrum(state, n)) });
                    if (store && n === sizes.at(-1)) {
                        store.turbulence ??= { width: n, height: n, dx: c.config.dx, times: [] };
                        store.turbulence.times.push({ t: gpu.time, ...fields(state, amount, n, n, c.config.dx, true) });
                    }
                }
            }
        } finally { gpu.destroy(); }
        rows.push({ n, series, spectra, massDrift, amountDrift, retries: gpu.counters.retries });
    }
    return rows;
}

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'dynamics', userAgent: navigator.userAgent, options: OPTIONS, errors: [] };
    const store = { timestamp: report.timestamp };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = kernels.sha256;
        const parts = document.querySelector('#parts').value.split(',');
        if (parts.includes('kh')) report.kelvinHelmholtz = await kelvinHelmholtz(device, kernels, [128, 256, 512], store);
        if (parts.includes('kh-fine')) report.kelvinHelmholtzFine = await kelvinHelmholtz(device, kernels, [1024], null);
        if (parts.includes('kh-lowfr')) report.kelvinHelmholtzLowFroude = await kelvinHelmholtz(device, kernels, [256, 512], null, 100);
        if (parts.includes('wake')) report.wake = await wake(device, kernels, [1, 0.5], store);
        if (parts.includes('wake-fine')) report.wakeFine = await wake(device, kernels, [0.25], null);
        if (parts.includes('exit')) report.vortexExit = await vortexExit(device, kernels);
        if (parts.includes('turbulence')) report.turbulence = await turbulence(device, kernels, [256, 512], store);
        device.destroy();
    } catch (error) { report.errors.push(error.stack ?? error.message); }
    window.__dynamics = report; window.__fields = store;
    const brief = JSON.parse(JSON.stringify(report, (k, v) => (['series', 'spectra', 'probe'].includes(k) ? undefined : v)));
    output.textContent = JSON.stringify(brief, null, 2);
    say(report.errors.length ? `Errors: ${report.errors.length}` : 'Done.');
});
