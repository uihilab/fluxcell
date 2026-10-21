import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { configuration, diagnostics, hydrographValue, hydrographGhost } from '../src/model.mjs';
import { scenario } from '../src/scenarios.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];
// Exact integral of a piecewise-linear series (constant outside) from its first time to T.
const exactIntegral = (series, T) => {
    let sum = 0;
    for (let k = 1; k < series.length; k++) {
        const [t0, v0] = series[k - 1], [t1, v1] = series[k];
        if (T <= t0) break;
        const b = Math.min(T, t1), vb = v0 + (v1 - v0) * (b - t0) / (t1 - t0);
        sum += (b - t0) * (v0 + vb) / 2;
    }
    const [tl, vl] = series.at(-1);
    return sum + Math.max(0, T - tl) * vl;
};
const volume = solver => diagnostics(solver.state, solver.config).volume;

for (const [label, Solver] of methods) {
    test(`${label}: a stage equal to the lake level keeps the lake exactly at rest`, () => {
        const { config, initial } = scenario('stage-lake', 13, 9), solver = new Solver(config, initial);
        for (let i = 0; i < 300; i++) solver.step(0.05);
        const error = Math.max(...solver.state.map((v, i) => i % 4 === 3 ? 0 : Math.abs(v - initial.state[i])));
        assert.ok(error < 1e-12, `max state error ${error}`);
        assert.ok(Math.abs(solver.boundaryOutflow) < 1e-12, `boundary volume ${solver.boundaryOutflow}`);
    });

    test(`${label}: a discharge boundary adds exactly the prescribed volume (constant and varying Q)`, () => {
        for (const series of [[[0, 3]], [[0, 0], [5, 4], [9, 1], [14, 2.5]]]) {
            const config = { width: 12, height: 7, dx: 2, dy: 3, boundaryX: 'wall', boundaryY: 'wall',
                hydrographs: [{ edge: 'west', start: 2, end: 6, kind: 'discharge', series }] };
            const n = 12 * 7, state = new Float32Array(4 * n), bed = new Float32Array(n);
            for (let i = 0; i < n; i++) { bed[i] = 0.05 * (i % 12); state[4 * i] = i % 12 > 6 ? 0 : 0.2; }
            const solver = new Solver(config, { state, bed });
            const v0 = volume(solver);
            while (solver.time < 20) solver.step(0.25);
            const expected = exactIntegral(series, solver.time); // the faces use the exact step mean
            assert.ok(Math.abs(-solver.boundaryOutflow - expected) <= 1e-12 * expected, `${-solver.boundaryOutflow} vs ${expected}`);
            assert.ok(Math.abs(volume(solver) - v0 - expected) <= 1e-11 * expected, `volume gain ${volume(solver) - v0} vs ${expected}`);
        }
    });

    test(`${label}: a delayed inflow onto a dry bed is admitted exactly, with a bounded first step`, () => {
        // EA Test 3 situation: dry start, no inflow until t = 5 s, then a ramp; long requested steps.
        const series = [[0, 0], [5, 0], [15, 65.5], [25, 65.5], [35, 0], [900, 0]];
        const n = 30 * 10, bed = new Float32Array(n), state = new Float32Array(4 * n);
        for (let i = 0; i < n; i++) bed[i] = 0.5 - 0.005 * 10 * (i % 30);
        const solver = new Solver({ width: 30, height: 10, dx: 10, dy: 10, boundaryX: 'wall', boundaryY: 'wall',
            hydrographs: [{ edge: 'west', kind: 'discharge', series }] }, { state, bed, manning: new Float32Array(n).fill(0.01) });
        const first = solver.step(2);
        const q = 65.5 / 100, bound = 0.4 / (3 * Math.cbrt(9.81 * q) / 10);
        assert.ok(first <= bound * (1 + 1e-12), `first dt ${first} exceeds the hydrograph bound ${bound}`);
        while (solver.time < 60) solver.step(2);
        const expected = exactIntegral(series, solver.time);
        assert.ok(Math.abs(-solver.boundaryOutflow - expected) <= 1e-12 * expected, `${-solver.boundaryOutflow} vs ${expected}`);
    });

    test(`${label}: discharge in and stage out reach the steady channel flow`, () => {
        const { config, initial } = scenario('hydrograph-channel', 40, 3), solver = new Solver(config, initial);
        for (let i = 0; i < 20000 && solver.time < 1800; i++) solver.step(1);
        const { width: w, height: h, dy } = solver.config, q = 2 / (h * dy);
        // Steady state: every x face carries the prescribed unit discharge. (Cell momenta differ
        // from face fluxes by the scheme's numerical diffusion on the friction slope.)
        for (let y = 0; y < h; y++) for (let x = 0; x <= w; x++) {
            const flux = solver.facesX[6 * (y * (w + 1) + x)];
            assert.ok(Math.abs(flux - q) < 1e-3 * q, `face mass flux at x face ${x}: ${flux} vs ${q}`);
        }
        for (let k = 0; k < solver.state.length; k += 4) assert.ok(Math.abs(solver.state[k + 2]) < 1e-9, 'no transverse flow');
        // Water surface in the last column sits at the downstream stage (0.5 m) to within 1 cm.
        const eta = solver.state[4 * (w - 1)] + solver.bed[w - 1] + solver.datum;
        assert.ok(Math.abs(eta - 0.5) < 0.01, `downstream surface ${eta}`);
        const residual = volume(solver) + solver.boundaryOutflow - diagnostics(initial.state, solver.config).volume;
        assert.ok(Math.abs(residual) < 1e-10 * volume(solver), `budget residual ${residual}`);
    });

    test(`${label}: a stage above a dry bed floods it with a closed budget and a bounded step`, () => {
        const n = 30 * 4, bed = new Float32Array(n), state = new Float32Array(4 * n);
        for (let i = 0; i < n; i++) bed[i] = 0.01 * (i % 30);
        const config = { width: 30, height: 4, dx: 1, dy: 1, boundaryX: 'outflow', boundaryY: 'wall',
            hydrographs: [{ edge: 'west', kind: 'stage', series: [[0, 1.5]] }] };
        const solver = new Solver(config, { state, bed });
        const first = solver.step(10);
        // Ghost depth 1.5 m: speed 2 sqrt(g h) bounds the first step although the domain is dry.
        assert.ok(first <= solver.config.cfl / (2 * Math.sqrt(9.81 * 1.5)) + 1e-12, `first dt ${first}`);
        for (let i = 0; i < 400; i++) solver.step(10);
        assert.ok(solver.state[4 * 15] > 0.1, 'the flood reaches mid-channel');
        const residual = volume(solver) + solver.boundaryOutflow;
        assert.ok(Math.abs(residual) < 1e-10 * volume(solver), `budget residual ${residual}`);
    });
}

// Backflow through a stage edge: water entering carries no tangential velocity (a copied one feeds
// boundary eddies back in and is unstable on fine grids).
test('stage backflow enters from rest (velocity head lost, no tangential velocity); outflow keeps both', () => {
    assert.deepEqual(hydrographGhost(1, 1, 1, 0.3, 0.2, 0, 9.81, -1), [1, 0.3, 0.2]); // east edge, outflow
    const head = 1 - 0.09 / (2 * 9.81); // inflow comes from rest at the stage
    assert.deepEqual(hydrographGhost(1, 1, 1, -0.3, 0.2, 0, 9.81, -1), [head, -0.3, 0]); // east edge, inflow
    assert.deepEqual(hydrographGhost(1, 1, 1, 0.3, 0.2, 0, 9.81, 1), [head, 0.3, 0]); // west edge, inflow
    for (const u of [-0.3, 0.3]) {
        const w = 12, h = 8, N = w * h, state = new Float64Array(4 * N);
        for (let i = 0; i < N; i++) state.set([1, u, 0.2, 0], 4 * i);
        const config = { width: w, height: h, dx: 1, dy: 1, hydrographs: [{ edge: 'east', kind: 'stage', series: [[0, 1]] }] };
        const solver = new MusclReferenceSolver(config, { state, bed: new Float64Array(N) });
        solver.step(0.05);
        for (let y = 0; y < h; y++) {
            const f = solver.facesX.subarray(6 * (y * (w + 1) + w), 6 * (y * (w + 1) + w) + 6);
            if (u < 0) assert.ok(f[0] < 0 && Math.abs(f[2]) < 1e-15, `inflow face ${y}: mass ${f[0]}, tangential flux ${f[2]}`);
            else assert.ok(Math.abs(f[2] - f[0] * 0.2) < 1e-12, `outflow face ${y}: tangential flux ${f[2]}`);
        }
    }
});

test('MUSCL and Rusanov reach the same steady channel flow', () => {
    const { config, initial } = scenario('hydrograph-channel', 40, 3);
    const a = new ReferenceSolver(config, initial), b = new MusclReferenceSolver(config, initial);
    for (let i = 0; i < 20000 && a.time < 1800; i++) a.step(1);
    for (let i = 0; i < 20000 && b.time < 1800; i++) b.step(1);
    for (let k = 0; k < a.state.length; k += 4) assert.ok(Math.abs(a.state[k] - b.state[k]) < 5e-3, `depth at cell ${k / 4}`);
});

test('hydrograph configurations are validated', () => {
    const base = { width: 8, height: 5, boundaryX: 'wall', boundaryY: 'periodic' };
    const bad = [
        [{ edge: 'south', kind: 'stage', series: [[0, 1]] }],                       // periodic edge
        [{ edge: 'west', kind: 'stage', series: [[0, 1]] }, { edge: 'west', start: 4, kind: 'stage', series: [[0, 1]] }], // overlap
        [{ edge: 'west', kind: 'discharge', series: [[0, 1], [0, 2]] }],             // times not increasing
        [{ edge: 'west', kind: 'discharge', series: [[0, -1]] }],                    // negative discharge
        [{ edge: 'top', kind: 'stage', series: [[0, 1]] }],                          // unknown edge
        [{ edge: 'east', kind: 'level', series: [[0, 1]] }],                         // unknown kind
        [{ edge: 'east', start: 3, end: 3, kind: 'stage', series: [[0, 1]] }],       // empty range
        [{ edge: 'east', kind: 'stage', series: [] }],                               // empty series
    ];
    for (const hydrographs of bad) assert.throws(() => configuration({ ...base, hydrographs }), RangeError);
    const ok = configuration({ ...base, hydrographs: [{ edge: 'west', end: 2, kind: 'stage', series: [[0, 1], [10, 2]] }] });
    assert.equal(configuration(ok).hydrographs.length, 1); // validated configurations round-trip
    assert.equal(hydrographValue([0, 10], [1, 2], 5), 1.5);
    assert.equal(hydrographValue([0, 10], [1, 2], -1), 1);
    assert.equal(hydrographValue([0, 10], [1, 2], 11), 2);
});
