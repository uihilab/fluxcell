import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { H_THIN } from '../src/model.mjs';
import { erf, advectionCase, diffusionCase, vortexCase, mixingLayerCase, meanAbsoluteError, turbulenceCase, energySpectrum,
    energyEnstrophy, modeAmplitude, shearLayerCase, wakeCase } from '../src/transport.mjs';

const sum = a => a.reduce((s, v) => s + v, 0);
const run = (solver, until) => { while (solver.time < until - 1e-12) solver.step(until - solver.time); return solver; };
const concentrations = s => Array.from(s.tracer, (hc, i) => (s.state[4 * i] > 0 ? hc / s.state[4 * i] : 0));
// A fixture with the tracer on and a concentration field.
function withTracer(name, width, height, concentration) {
    const { config, initial } = scenario(name, width, height);
    const c = new Float64Array(width * height);
    for (let i = 0; i < c.length; i++) c[i] = concentration(i % width, Math.floor(i / width));
    return { config: { ...config, tracer: true }, initial: { ...initial, concentration: c } };
}

test('erf matches tabulated values', () => {
    for (const [x, v] of [[0.5, 0.5204998778130465], [1, 0.8427007929497149], [2, 0.9953222650189527], [3.5, 0.9999992569016276]]) {
        assert.ok(Math.abs(erf(x) - v) < 1e-14 && Math.abs(erf(-x) + v) < 1e-14, `erf(${x})`);
    }
});

for (const [label, Solver] of [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]]) {
    test(`${label}: a uniform concentration stays uniform through a dam break onto dry ground`, () => {
        const { config, initial } = withTracer('dry-dam', 24, 9, () => 0.7);
        const solver = run(new Solver(config, initial), 3);
        let worst = 0;
        concentrations(solver).forEach((c, i) => { if (solver.state[4 * i] > H_THIN) worst = Math.max(worst, Math.abs(c - 0.7)); });
        assert.ok(worst < 1e-12, `max |c - 0.7| = ${worst}`);
    });

    test(`${label}: tracer amount is conserved and bounded with obstacles and wetting`, () => {
        const { config, initial } = withTracer('obstacle', 21, 15, (x, y) => ((x * 7 + y * 3) % 11) / 10);
        const solver = new Solver(config, initial), amount0 = sum(solver.tracer);
        run(solver, 4);
        assert.ok(Math.abs(sum(solver.tracer) - amount0) < 1e-12 * amount0, `amount ${sum(solver.tracer)} vs ${amount0}`);
        concentrations(solver).forEach((c, i) => {
            if (solver.state[4 * i] > H_THIN) assert.ok(c >= -1e-12 && c <= 1 + 1e-12, `c = ${c} at cell ${i}`);
        });
    });

    test(`${label}: inflow carries its concentration and the open-boundary budget closes`, () => {
        const { config, initial } = scenario('hydrograph-channel', 40, 3);
        config.tracer = true;
        config.hydrographs = config.hydrographs.map(s => (s.kind === 'discharge' ? { ...s, concentration: 2 } : s));
        const solver = run(new Solver(config, initial), 600);
        const residual = sum(solver.tracer) * config.dx * config.dy + solver.tracerOutflow;
        // Inflow enters as negative outflow: it equals 2 times the inflow volume, which is exact.
        const inflowVolume = 2 * 20 / 2 + 2 * (600 - 20);
        assert.ok(Math.abs(residual) < 1e-9 * 2 * inflowVolume, `tracer budget residual ${residual}`);
        concentrations(solver).forEach(c => assert.ok(c >= -1e-12 && c <= 2 + 1e-12));
    });

    test(`${label}: source water brings its concentration`, () => {
        const { config, initial } = scenario('sources', 13, 11);
        const n = 13 * 11, sourceConcentration = new Float64Array(n).fill(3);
        const solver = run(new Solver({ ...config, tracer: true }, { ...initial, sourceConcentration }), 20);
        const amount = sum(solver.tracer) * config.dx * config.dy;
        assert.ok(Math.abs(amount - solver.tracerSourceAmount) < 1e-10 * solver.tracerSourceAmount, `${amount} vs ${solver.tracerSourceAmount}`);
        // Rain (c = 0) dilutes the source water.
        concentrations(solver).forEach((c, i) => { if (solver.state[4 * i] > H_THIN) assert.ok(c >= -1e-12 && c <= 3 + 1e-12); });
    });

    test(`${label}: a lake at rest leaves any tracer field unchanged`, () => {
        const { config, initial } = withTracer('lake', 17, 13, (x, y) => (x + 2 * y) % 5 / 4);
        const solver = new Solver(config, initial), before = solver.tracer.slice();
        run(solver, 5);
        assert.ok(solver.tracer.every((v, i) => Math.abs(v - before[i]) < 1e-14), 'tracer moved at rest');
    });
}

test('advection of a Gaussian converges at second order (MUSCL) and first order (Rusanov)', () => {
    const errors = { MUSCL: [], Rusanov: [] };
    for (const n of [64, 128, 256]) {
        for (const [label, Solver] of [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]]) {
            const { config, initial, exact } = advectionCase(n);
            const solver = run(new Solver(config, initial), 2);
            errors[label].push(meanAbsoluteError(concentrations(solver), exact(solver.time).concentration));
        }
    }
    const order = e => Math.log2(e.at(-2) / e.at(-1));
    assert.ok(order(errors.MUSCL) > 1.6, `MUSCL order ${order(errors.MUSCL)} (${errors.MUSCL})`);
    assert.ok(order(errors.Rusanov) > 0.7 && order(errors.Rusanov) < 1.3, `Rusanov order ${order(errors.Rusanov)}`);
    assert.ok(errors.MUSCL.at(-1) < errors.Rusanov.at(-1) / 10, 'MUSCL is far more accurate');
});

test('a square pulse stays within [0, 1] (MUSCL)', () => {
    const { config, initial } = advectionCase(128, 'square');
    const solver = run(new MusclReferenceSolver(config, initial), 10);
    const c = concentrations(solver);
    assert.ok(Math.min(...c) >= -1e-13 && Math.max(...c) <= 1 + 1e-13, `range ${Math.min(...c)} ${Math.max(...c)}`);
});

test('diffusion of a Gaussian in still water matches the exact solution at second order', () => {
    const errors = [];
    for (const n of [32, 64, 128]) {
        const { config, initial, exact } = diffusionCase(n, 0.05);
        const solver = run(new MusclReferenceSolver(config, initial), 10);
        errors.push(meanAbsoluteError(concentrations(solver), exact(solver.time).concentration));
    }
    const order = Math.log2(errors[1] / errors[2]);
    assert.ok(order > 1.7, `order ${order} (${errors})`);
});

test('the cyclostrophic vortex stays steady and winds the tracer front as predicted', () => {
    const { config, initial, exact } = vortexCase(64);
    const solver = run(new MusclReferenceSolver(config, initial), 20);
    const ex = exact(solver.time);
    const depthError = meanAbsoluteError(Array.from({ length: ex.depth.length }, (_, i) => solver.state[4 * i]), ex.depth);
    const cError = meanAbsoluteError(concentrations(solver), ex.concentration);
    assert.ok(depthError < 2e-4, `depth L1 ${depthError}`);
    assert.ok(cError < 0.02, `concentration L1 ${cError}`);
});

test('the mixing layer approaches the erf profile', () => {
    const { config, initial, exact } = mixingLayerCase(40);
    const solver = run(new MusclReferenceSolver(config, initial), 900);
    const error = meanAbsoluteError(concentrations(solver), exact().concentration);
    assert.ok(error < 0.03, `L1 ${error}`);
});

test('tracer options are validated', () => {
    const base = { width: 4, height: 3 }, initial = { state: new Float64Array(48), bed: new Float64Array(12) };
    assert.throws(() => new ReferenceSolver({ ...base, diffusivity: 1 }, initial), /requires tracer/);
    assert.throws(() => new ReferenceSolver(base, { ...initial, concentration: new Float64Array(12) }), /requires tracer/);
    assert.throws(() => new ReferenceSolver({ ...base, tracer: true }, { ...initial, concentration: new Float64Array(5) }), /length/);
    assert.throws(() => new ReferenceSolver({ ...base, tracer: true }, { ...initial, concentration: new Float64Array(12).fill(-1) }), /Invalid concentration/);
    assert.throws(() => new ReferenceSolver({ ...base, tracer: true, hydrographs: [{ edge: 'west', kind: 'stage', series: [[0, 1]], concentration: -1 }] }, initial), /concentration/);
});

test('spectrum, energy and mode diagnostics are exact on known fields', () => {
    // Parseval: the shell spectrum sums to the mean kinetic energy.
    const { initial } = turbulenceCase(64, { urms: 0.5 });
    const E = energySpectrum(initial.state, 64), { energy } = energyEnstrophy(initial.state, 64, 100 / 64);
    assert.ok(Math.abs(E.reduce((a, b) => a + b, 0) - energy) < 1e-12 && Math.abs(energy - 0.125) < 1e-12);
    // Energy sits in the forced shell (k = 6..10) only.
    E.forEach((e, k) => { if (k < 6 || k > 10) assert.ok(e < 1e-20, `E(${k}) = ${e}`); });
    // A pure transverse sine of amplitude a in every row has mode amplitude a / sqrt(2) * sqrt(2) = a.
    const n = 32, state = new Float64Array(4 * n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) state.set([2, 0, 2 * 0.3 * Math.sin(2 * Math.PI * 3 * (x + 0.5) / n), 0], 4 * (y * n + x));
    assert.ok(Math.abs(modeAmplitude(state, n, 3) - 0.3) < 1e-12 && modeAmplitude(state, n, 2) < 1e-12);
    // The shear layer and the wake are well posed inputs for the solvers.
    const kh = shearLayerCase(32), wake = wakeCase(2);
    for (const c of [kh, wake]) new MusclReferenceSolver(c.config, c.initial);
    assert.ok(Math.abs(kh.delta - 0.4446 * 100 / (8 * Math.PI)) < 1e-12);
});
