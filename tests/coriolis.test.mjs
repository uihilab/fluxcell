import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { configuration, coriolisRotate, diagnostics } from '../src/model.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];

test('coriolis option is validated and the rotation is exact', () => {
    for (const coriolis of [NaN, Infinity, 2e-3, -2e-3, 'f']) assert.throws(() => configuration({ width: 2, height: 2, coriolis }), RangeError);
    assert.equal(configuration({ width: 2, height: 2 }).coriolis, 0);
    // du/dt = f v, dv/dt = -f u: u = U cos ft + V sin ft, v = -U sin ft + V cos ft; steps compose.
    let q = [0.375, -0.125];
    for (let k = 0; k < 1000; k++) q = coriolisRotate(1e-4, 7.5, ...q);
    const ft = 1e-4 * 7500;
    assert.ok(Math.abs(q[0] - (0.375 * Math.cos(ft) - 0.125 * Math.sin(ft))) < 1e-12);
    assert.ok(Math.abs(q[1] - (-0.375 * Math.sin(ft) - 0.125 * Math.cos(ft))) < 1e-12);
    assert.ok(Math.abs(Math.hypot(...q) - Math.hypot(0.375, -0.125)) < 1e-13, 'rotation conserves |q| (to rounding over 1000 steps)');
});

for (const [label, Solver] of methods) {
    test(`${label}: a uniform current turns in an exact inertial circle`, () => {
        const { config, initial } = scenario('coriolis-uniform', 9, 7), solver = new Solver(config, initial);
        while (solver.time < 3000) solver.step(Math.min(5, 3000 - solver.time));
        const ft = 1e-3 * solver.time;
        for (let k = 0; k < solver.state.length; k += 4) {
            assert.ok(Math.abs(solver.state[k] - 1.25) < 1e-12);
            assert.ok(Math.abs(solver.state[k + 1] - (0.375 * Math.cos(ft) - 0.125 * Math.sin(ft))) < 1e-12);
            assert.ok(Math.abs(solver.state[k + 2] - (-0.375 * Math.sin(ft) - 0.125 * Math.cos(ft))) < 1e-12);
        }
    });

    // Rusanov is not tested here: its first-order mass diffusion (about c dx / 2 across the jet)
    // flattens the jet within the run at these resolutions, as expected of that scheme.
    if (label === 'MUSCL') test(`${label}: a geostrophic jet stays in balance, with the error falling under refinement`, () => {
        // Flat bed, periodic in x, walls in y: h(y) = h0 + a tanh((y - L/2)/w), u = -(g/f) dh/dy, v = 0
        // is an exact steady solution. Initialized with cell averages of h and q, it drifts only by
        // discretization error.
        const run = n => {
            const L = 10000, dx = L / n, f = 1e-4, g = 9.81, h0 = 10, a = 0.05, w = L / 8, nx = 4;
            const H = y => h0 + a * w / dx * (Math.log(Math.cosh((y + dx - L / 2) / w)) - Math.log(Math.cosh((y - L / 2) / w)));
            const state = new Float64Array(4 * nx * n), bed = new Float64Array(nx * n);
            for (let y = 0; y < n; y++) {
                const hAvg = H(y * dx), hFace = yy => h0 + a * Math.tanh((yy - L / 2) / w);
                // q_x = h u = -(g/f) h dh/dy = -(g/2f) d(h²)/dy, averaged over the cell.
                const qAvg = -(g / (2 * f)) * (hFace((y + 1) * dx) ** 2 - hFace(y * dx) ** 2) / dx;
                for (let x = 0; x < nx; x++) state.set([hAvg, qAvg, 0, 0], 4 * (y * nx + x));
            }
            const solver = new Solver({ width: nx, height: n, dx, dy: dx, boundaryX: 'periodic', boundaryY: 'wall', coriolis: f }, { state, bed });
            const volume0 = diagnostics(solver.state, solver.config).volume;
            while (solver.time < 4e4) solver.step(4e4 - solver.time);
            let err = 0;
            for (let k = 0; k < state.length; k += 4) err = Math.max(err, Math.abs(solver.state[k] - state[k]));
            assert.ok(Math.abs(diagnostics(solver.state, solver.config).volume - volume0) < 1e-11 * volume0);
            return err / a;
        };
        const coarse = run(32), fine = run(128);
        assert.ok(fine < 0.01, `relative depth error ${fine}`);
        assert.ok(fine < 0.5 * coarse, `error must fall with dx: ${coarse} -> ${fine}`);
    });

    test(`${label}: a rotating basin with an obstacle keeps depths nonnegative and conserves water`, () => {
        const { config, initial } = scenario('coriolis-basin', 17, 13), solver = new Solver(config, initial);
        const volume0 = diagnostics(initial.state, config).volume;
        for (let k = 0; k < 600; k++) solver.step(0.05);
        const stats = diagnostics(solver.state, solver.config);
        assert.ok(stats.minDepth >= 0 && stats.maxSpeed > 0);
        assert.ok(Math.abs(stats.volume - volume0) <= 1e-12 * volume0);
    });
}
