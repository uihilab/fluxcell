import { compileKernels, createSolver } from '../../src/webgpu.mjs';
import { MusclReferenceSolver } from '../../src/muscl-reference.mjs';
import { scenario } from '../../src/scenarios.mjs';
import { fuzzScenario } from '../../src/fuzz.mjs';
import { H_THIN, tracerBounds, initialData, configuration, settledConcentration } from '../../src/model.mjs';
import { advectionCase, diffusionCase, vortexCase, mixingLayerCase, viscousShearCase, meanAbsoluteError } from '../../src/transport.mjs';

// GPU verification of scalar transport in the fused kernel: equivalence with the f64 MUSCL
// reference, conservation, bounds and uniform-concentration preservation, convergence to exact
// solutions, bitwise batch invariance, and randomized stress with rollback.
const status = document.querySelector('#status'), output = document.querySelector('#output');
const say = text => { status.textContent = text; };
const sumOf = a => { let s = 0; for (const v of a) s += v; return s; };
const ratio = (a, b) => Array.from(a, (v, i) => (b[4 * i] > 0 ? v / b[4 * i] : 0));
const order = e => e.slice(1).map((v, k) => Math.log2(e[k] / v));

async function advanceTo(solver, until, batch = 64) {
    if (until > solver.time) await solver.advance(until - solver.time, { batch });
}

// Bound violations of the concentration in cells deeper than H_THIN, relative to the range.
function boundsCheck(state, amount, [lo, hi]) {
    let worst = 0;
    const range = Math.max(hi - lo, Math.abs(hi), 1e-30);
    for (let i = 0; i < amount.length; i++) {
        const h = state[4 * i];
        if (!(h > H_THIN)) continue;
        const c = amount[i] / h;
        worst = Math.max(worst, (lo - c) / range, (c - hi) / range);
    }
    return Math.max(0, worst);
}

// Tracer budget: final amount + outflow + deposited - initial - sources, relative to the largest
// term. The kernel does not accumulate settled mass; with settling, the f64 reference's deposit
// (equivalent run) closes the GPU budget.
async function tracerBudget(solver, config, amount0, deposited = 0) {
    const amount = await solver.readTracer(), outflow = await solver.readBoundaryOutflow({ tracer: true });
    const total = sumOf(amount) * config.dx * config.dy;
    const residual = total + outflow + deposited - amount0 - solver.tracerSourceAmount;
    return { residual, relative: Math.abs(residual) / Math.max(amount0 + solver.tracerSourceAmount, total, Math.abs(outflow), 1e-30) };
}

const FIXTURES = ['dam', 'dry-dam', 'obstacle', 'pulse', 'sources', 'lake', 'shoreline', 'stage-lake', 'hydrograph-channel', 'open-dam', 'outflow-drain'];
const CLOSURE = { smagorinsky: 0.2, bedTurbulence: 0.067, eddyViscosity: 0.002 };
function fixture(name, { diffusivity = 0, uniform = null, closure = null, settling = null } = {}) {
    const { config, initial } = scenario(name, 37, 29);
    if (closure) Object.assign(config, closure);
    if (settling) config.settling = settling;
    const n = config.width * config.height, concentration = new Float32Array(n), sourceConcentration = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const x = i % config.width, y = Math.floor(i / config.width);
        concentration[i] = uniform ?? ((x * 7 + y * 3) % 11) / 10;
        sourceConcentration[i] = uniform ?? 1.5;
    }
    const hydrographs = (config.hydrographs ?? []).map(h => ({ ...h, concentration: uniform ?? 2 }));
    return { config: { ...config, tracer: true, diffusivity, hydrographs },
        initial: { ...initial, concentration, sourceConcentration } };
}

async function equivalence(device, kernels, name, options, steps = 300) {
    const { config, initial } = fixture(name, options);
    const gpu = await createSolver(device, config, initial, kernels), cpu = new MusclReferenceSolver(config, initial);
    const amount0 = sumOf(cpu.tracer) * config.dx * config.dy;
    try {
        let done = 0;
        while (done < steps) {
            const { dts } = await gpu.advance(0.05 * (steps - done), { maxDt: 0.05, batch: 64, maxSteps: steps - done });
            for (const dt of dts) cpu.step(dt);
            done += dts.length;
        }
        const state = await gpu.readState(), amount = await gpu.readTracer();
        let stateDiff = 0, amountDiff = 0, concDiff = 0, scale = Math.max(...cpu.bounds.map(Math.abs), 1e-30);
        for (let k = 0; k < state.length; k++) if (k % 4 !== 3) stateDiff = Math.max(stateDiff, Math.abs(state[k] - cpu.state[k]));
        for (let i = 0; i < amount.length; i++) {
            amountDiff = Math.max(amountDiff, Math.abs(amount[i] - cpu.tracer[i]) / scale);
            if (cpu.state[4 * i] > 1e-3) concDiff = Math.max(concDiff, Math.abs(amount[i] / state[4 * i] - cpu.tracer[i] / cpu.state[4 * i]) / scale);
        }
        const budget = await tracerBudget(gpu, config, amount0, cpu.tracerDeposited);
        return { name, diffusivity: options.diffusivity ?? 0, settling: options.settling ?? null, deposited: cpu.tracerDeposited, steps: done, time: gpu.time, stateDiff, amountDiff, concDiff,
            budgetRelative: budget.relative, boundViolation: boundsCheck(state, amount, cpu.bounds), retries: gpu.counters.retries };
    } finally { gpu.destroy(); }
}

async function uniformity(device, kernels, name) {
    const { config, initial } = fixture(name, { uniform: 0.3 });
    if (config.rain > 0) return null; // rain carries no tracer and dilutes
    const gpu = await createSolver(device, config, initial, kernels);
    try {
        await gpu.advance(1e6, { maxDt: 0.05, batch: 64, maxSteps: 300 });
        const state = await gpu.readState(), amount = await gpu.readTracer();
        let worst = 0;
        for (let i = 0; i < amount.length; i++) if (state[4 * i] > H_THIN) worst = Math.max(worst, Math.abs(amount[i] / state[4 * i] - Math.fround(0.3)) / 0.3);
        return { name, maxRelativeDeviation: worst };
    } finally { gpu.destroy(); }
}

async function convergence(device, kernels) {
    const out = {};
    const series = async (label, sizes, make, until, measure) => {
        const rows = [];
        for (const n of sizes) {
            say(`${label} n = ${n}`);
            const c = make(n), gpu = await createSolver(device, c.config, c.initial, kernels);
            try {
                await advanceTo(gpu, until, 256);
                rows.push({ n, time: gpu.time, ...measure(c, await gpu.readState(), await gpu.readTracer(), gpu) });
            } finally { gpu.destroy(); }
        }
        out[label] = rows;
    };
    await series('advection-gaussian', [64, 128, 256, 512], n => advectionCase(n), 2,
        (c, state, amount, gpu) => ({ l1: meanAbsoluteError(ratio(amount, state), c.exact(gpu.time).concentration) }));
    await series('advection-square', [256], n => advectionCase(n, 'square'), 10, (c, state, amount, gpu) => {
        const conc = ratio(amount, state);
        return { l1: meanAbsoluteError(conc, c.exact(gpu.time).concentration), min: Math.min(...conc), max: Math.max(...conc) };
    });
    await series('diffusion', [32, 64, 128, 256], n => diffusionCase(n, 0.05), 10,
        (c, state, amount, gpu) => ({ l1: meanAbsoluteError(ratio(amount, state), c.exact(gpu.time).concentration) }));
    await series('vortex', [64, 128, 256], n => vortexCase(n), 60, (c, state, amount, gpu) => {
        const ex = c.exact(gpu.time), depth = Array.from(ex.depth, (_, i) => state[4 * i]);
        let peak = 0;
        for (let i = 0; i < ex.depth.length; i++) peak = Math.max(peak, Math.hypot(state[4 * i + 1], state[4 * i + 2]) / state[4 * i]);
        return { depthL1: meanAbsoluteError(depth, ex.depth), concentrationL1: meanAbsoluteError(ratio(amount, state), ex.concentration),
            peakSpeedRatio: peak / c.V };
    });
    await series('mixing-layer', [50, 100, 200, 400], n => mixingLayerCase(n), 1200,
        (c, state, amount) => ({ l1: meanAbsoluteError(ratio(amount, state), c.exact().concentration) }));
    for (const rows of Object.values(out)) {
        const key = rows[0].l1 !== undefined ? 'l1' : 'concentrationL1';
        if (rows.length > 1) rows.forEach((r, k) => { if (k) r.order = Math.log2(rows[k - 1][key] / r[key]); });
    }
    return out;
}

// Settling in a lake at rest: no transport, so each cell follows the exact local solution
// c(t) of dc/dt = -w_s(c) c / h from a uniform start (model.mjs: settledConcentration over t).
async function settlingExact(device, kernels, settling) {
    const { config, initial } = fixture('lake', { uniform: 0.8, settling });
    const gpu = await createSolver(device, config, initial, kernels);
    try {
        await gpu.advance(1e6, { maxDt: 0.05, batch: 64, maxSteps: 400 });
        const state = await gpu.readState(), amount = await gpu.readTracer();
        let worst = 0, lowest = Infinity;
        for (let i = 0; i < amount.length; i++) {
            const h = initial.state[4 * i];
            if (!(h > 0.05)) continue;
            const exact = settledConcentration(configuration(config).settling, Math.fround(0.8), h, gpu.time);
            worst = Math.max(worst, Math.abs(amount[i] / state[4 * i] - exact) / Math.fround(0.8));
            lowest = Math.min(lowest, exact / Math.fround(0.8));
        }
        return { settling, time: gpu.time, maxRelativeError: worst, smallestRemainingFraction: lowest };
    } finally { gpu.destroy(); }
}

// The same interval in batches of different sizes must give bitwise identical fields.
async function batchInvariance(device, kernels, name, options) {
    const { config, initial } = fixture(name, options);
    const results = [];
    for (const batch of [1, 13, 64, 256]) {
        const gpu = await createSolver(device, config, initial, kernels);
        try {
            let steps = 0;
            while (steps < 256) steps += (await gpu.advance(1e6, { maxDt: 0.05, batch, maxSteps: 256 - steps })).steps;
            results.push({ batch, state: await gpu.readState(), amount: await gpu.readTracer(), time: gpu.time });
        } finally { gpu.destroy(); }
    }
    const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
    return { name, diffusivity: options.diffusivity ?? 0, identical: results.every(r => same(r.state, results[0].state)
        && same(r.amount, results[0].amount) && r.time === results[0].time) };
}

// Fault injection rejects every step longer than faultDt after faultTime, so batches fail mid-way,
// valid prefixes are replayed and single steps retried at half dt. The accepted run must equal,
// bit for bit, a plain step-by-step run with the same dts (state, tracer, both boundary
// accumulators, time); encode/resolve must recover the same way.
async function rollback(device, kernels, name, options, until = 2) {
    const fixed = fixture(name, options), tracer = options.tracer !== false; // options.closure passes through
    const config = { ...fixed.config, tracer, diffusivity: tracer ? fixed.config.diffusivity : 0 };
    const initial = tracer ? fixed.initial : { ...fixed.initial, concentration: undefined, sourceConcentration: undefined };
    const probe = await createSolver(device, config, initial, kernels);
    const dt0 = await probe.step(1e6);
    probe.destroy();
    const faultInjection = { dt: 0.6 * dt0, time: 10 * dt0 };
    const faulty = await createSolver(device, { ...config, faultInjection }, initial, kernels);
    const plain = await createSolver(device, config, initial, kernels);
    try {
        const { dts } = await faulty.advance(Math.max(until, 40 * dt0), { batch: 64 });
        let mismatchedDt = 0;
        for (const dt of dts) if ((await plain.step(dt)) !== dt) mismatchedDt++;
        const read = async s => ({ state: await s.readState(), amount: tracer ? await s.readTracer() : [],
            water: await s.readBoundaryOutflow(), tracer: await s.readBoundaryOutflow({ tracer: true }), time: s.time });
        const [a, b] = [await read(faulty), await read(plain)];
        const same = (x, y) => x.length === y.length && x.every((v, i) => Object.is(v, y[i]));
        // encode/resolve: one batch recorded into a caller encoder, recovered after the fault.
        const encoded = await createSolver(device, { ...config, faultInjection }, initial, kernels);
        let recovered = false, encodedMatches = false;
        try {
            const encoder = device.createCommandEncoder();
            encoded.encode(encoder, { duration: 40 * dt0, steps: 64 });
            device.queue.submit([encoder.finish()]);
            const result = await encoded.resolve();
            recovered = result.recovered;
            const reference = await createSolver(device, config, initial, kernels);
            try {
                for (const dt of result.dts) await reference.step(dt);
                encodedMatches = same(await encoded.readState(), await reference.readState())
                    && (!tracer || same(await encoded.readTracer(), await reference.readTracer()));
            } finally { reference.destroy(); }
        } finally { encoded.destroy(); }
        return { name, tracer, diffusivity: options.diffusivity ?? 0, steps: dts.length, retries: faulty.counters.retries,
            smallestDtRatio: Math.min(...dts) / dt0, mismatchedDt,
            identical: same(a.state, b.state) && same(a.amount, b.amount) && a.water === b.water && a.tracer === b.tracer && a.time === b.time,
            encodeRecovered: recovered, encodeIdentical: encodedMatches };
    } finally { faulty.destroy(); plain.destroy(); }
}

// Viscous shear layer with a constant eddy viscosity against its exact erf profile.
async function viscousConvergence(device, kernels) {
    const rows = [];
    for (const n of [32, 64, 128, 256]) {
        say(`viscous shear n = ${n}`);
        const c = viscousShearCase(n), gpu = await createSolver(device, c.config, c.initial, kernels);
        try {
            await advanceTo(gpu, 20, 256);
            const state = await gpu.readState(), u = Array.from({ length: state.length / 4 }, (_, i) => state[4 * i + 1] / state[4 * i]);
            rows.push({ n, time: gpu.time, l1: meanAbsoluteError(u, c.exact(gpu.time).velocity) });
        } finally { gpu.destroy(); }
    }
    rows.forEach((r, k) => { if (k) r.order = Math.log2(rows[k - 1].l1 / r.l1); });
    return rows;
}

async function stress(device, kernels, seeds, steps, closure = null) {
    const rows = [];
    for (const seed of seeds) {
        say(`stress seed ${seed}`);
        const c = fuzzScenario(seed, { steps, tracer: true }), row = { seed, description: c.description };
        if (closure) Object.assign(c.config, closure);
        let gpu;
        try {
            const config = configuration(c.config), data = initialData(config, c.initial, Float32Array);
            const bounds = tracerBounds(config, data);
            const amount0 = sumOf(data.tracer) * config.dx * config.dy;
            gpu = await createSolver(device, c.config, c.initial, kernels);
            const { steps: taken } = await gpu.advance(1e7, { maxDt: 10, batch: 64, maxSteps: steps });
            const state = await gpu.readState(), amount = await gpu.readTracer();
            const budget = await tracerBudget(gpu, config, amount0);
            Object.assign(row, { steps: taken, retries: gpu.counters.retries, budgetRelative: budget.relative,
                budgetAbsolutePerArea: Math.abs(budget.residual) / (config.width * config.height * config.dx * config.dy),
                boundViolation: boundsCheck(state, amount, bounds), finite: amount.every(Number.isFinite) });
        } catch (error) { row.fatal = error.message; }
        finally { gpu?.destroy(); }
        rows.push(row);
    }
    return rows;
}

document.querySelector('#run').addEventListener('click', async () => {
    const report = { timestamp: new Date().toISOString(), kind: 'transport', userAgent: navigator.userAgent, errors: [] };
    try {
        const adapter = await navigator.gpu.requestAdapter();
        report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture };
        const device = await adapter.requestDevice();
        device.addEventListener('uncapturederror', event => report.errors.push(event.error.message));
        const kernels = await compileKernels(device, 'muscl-fused');
        report.shaderSha256 = kernels.sha256;
        const parts = document.querySelector('#parts').value.split(',');
        if (parts.includes('equivalence')) {
            report.equivalence = [];
            for (const name of FIXTURES) {
                say(`equivalence ${name}`);
                report.equivalence.push(await equivalence(device, kernels, name, {}));
            }
            for (const name of ['dam', 'pulse', 'obstacle', 'hydrograph-channel']) {
                say(`equivalence ${name} with diffusion`);
                report.equivalence.push(await equivalence(device, kernels, name, { diffusivity: 0.05 }));
            }
            report.uniformity = [];
            for (const name of FIXTURES) { say(`uniform ${name}`); const r = await uniformity(device, kernels, name); if (r) report.uniformity.push(r); }
        }
        if (parts.includes('batch')) {
            report.batchInvariance = [];
            for (const [name, options] of [['pulse', { diffusivity: 0.05 }], ['hydrograph-channel', {}], ['obstacle', {}]]) {
                say(`batch invariance ${name}`);
                report.batchInvariance.push(await batchInvariance(device, kernels, name, options));
            }
        }
        if (parts.includes('rollback')) {
            report.rollback = [];
            for (const [name, options] of [['pulse', { diffusivity: 0.05 }], ['hydrograph-channel', {}], ['dry-dam', {}], ['outflow-drain', {}], ['obstacle', { tracer: false }]]) {
                say(`rollback ${name}`);
                report.rollback.push(await rollback(device, kernels, name, options));
            }
        }
        if (parts.includes('settling')) {
            // Tracer settling: exact solution at rest, equivalence (with the budget closed by the
            // reference's deposit), batch invariance and forced rollback, constant and flocculating.
            const SETTLING = [{ velocity: 2e-3, exponent: 0 }, { velocity: 1.5e-3, exponent: 4 / 3 }];
            report.settling = { exact: [], equivalence: [], batchInvariance: [], rollback: [] };
            for (const settling of SETTLING) {
                say(`settling exact ${settling.exponent}`);
                report.settling.exact.push(await settlingExact(device, kernels, settling));
                for (const name of ['dam', 'pulse', 'obstacle', 'sources', 'hydrograph-channel', 'shoreline', 'outflow-drain']) {
                    say(`settling equivalence ${name}`);
                    report.settling.equivalence.push(await equivalence(device, kernels, name, { settling, diffusivity: name === 'pulse' ? 0.05 : 0 }));
                }
                report.settling.batchInvariance.push(await batchInvariance(device, kernels, 'hydrograph-channel', { settling }));
                report.settling.rollback.push(await rollback(device, kernels, 'pulse', { settling }));
            }
        }
        if (parts.includes('convergence')) report.convergence = await convergence(device, kernels);
        if (parts.includes('closure')) {
            // Turbulence closure: equivalence with the f64 MUSCL reference, exact viscous shear layer,
            // forced rollback, and randomized stress.
            report.closure = { settings: CLOSURE, equivalence: [], rollback: [] };
            for (const name of ['dam', 'pulse', 'obstacle', 'friction', 'hydrograph-channel', 'open-dam', 'dry-dam']) {
                say(`closure equivalence ${name}`);
                report.closure.equivalence.push(await equivalence(device, kernels, name, { closure: CLOSURE }));
            }
            report.closure.viscousShear = await viscousConvergence(device, kernels);
            for (const name of ['pulse', 'friction']) {
                say(`closure rollback ${name}`);
                report.closure.rollback.push(await rollback(device, kernels, name, { closure: CLOSURE }));
            }
            report.closure.stress = await stress(device, kernels, Array.from({ length: 200 }, (_, i) => i + 1001), 400, { smagorinsky: 0.15, bedTurbulence: 0.067 });
        }
        if (parts.includes('stress')) {
            const count = Number(document.querySelector('#count').value);
            report.stress = await stress(device, kernels, Array.from({ length: count }, (_, i) => i + 1), 400);
        }
        device.destroy();
    } catch (error) { report.errors.push(error.stack ?? error.message); }
    window.__transport = report;
    output.textContent = JSON.stringify(report, (k, v) => (typeof v === 'number' && !Number.isInteger(v) ? Number(v.toPrecision(4)) : v), 2).slice(0, 20000);
    say(report.errors.length ? `Errors: ${report.errors.length}` : 'Done.');
});
