import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { configuration, windStress, windDragCoefficient, diagnostics } from '../src/model.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];
const run = (solver, steps, maxDt = 0.05) => { for (let i = 0; i < steps; i++) solver.step(maxDt); return solver; };

test('wind options are validated', () => {
    const base = { width: 2, height: 2 };
    for (const wind of [[1], [1, 2, 3], [NaN, 0], [0, Infinity], [100, 1], 'east']) {
        assert.throws(() => configuration({ ...base, wind }), RangeError, JSON.stringify(wind));
    }
    for (const windDrag of [0, -1e-3, 0.02, 'large', NaN]) assert.throws(() => configuration({ ...base, windDrag }), RangeError);
    assert.deepEqual(windStress(configuration(base)), [0, 0]);
});

test('wind stress follows Wu (1982) or a constant drag coefficient', () => {
    // Wu (1982): C_d = (0.8 + 0.065 W) 1e-3; 10 m/s gives 1.45e-3 and tau / rho_w = 1.225e-3 C_d W².
    const p = configuration({ width: 1, height: 1, wind: [6, 8] });
    assert.ok(Math.abs(windDragCoefficient(p, 10) - 1.45e-3) < 1e-15);
    const [sx, sy] = windStress(p);
    assert.ok(Math.abs(Math.hypot(sx, sy) - 1.225e-3 * 1.45e-3 * 100) < 1e-15);
    assert.ok(Math.abs(sx / sy - 6 / 8) < 1e-12, 'stress is along the wind');
    const constant = windStress(configuration({ width: 1, height: 1, wind: [0, -20], windDrag: 2e-3 }));
    assert.ok(Math.abs(constant[0]) < 1e-18 && Math.abs(constant[1] + 1.225e-3 * 2e-3 * 400) < 1e-15);
});

for (const [label, Solver] of methods) {
    test(`${label}: uniform wind on a periodic domain accelerates the water as q = tau t`, () => {
        const { config, initial } = scenario('wind-uniform', 9, 7), solver = run(new Solver(config, initial), 400);
        const [sx, sy] = windStress(solver.config);
        for (let k = 0; k < solver.state.length; k += 4) {
            assert.ok(Math.abs(solver.state[k] - 1.25) < 1e-12);
            assert.ok(Math.abs(solver.state[k + 1] - sx * solver.time) < 1e-12, `${solver.state[k + 1]} vs ${sx * solver.time}`);
            assert.ok(Math.abs(solver.state[k + 2] - sy * solver.time) < 1e-12);
        }
    });

    test(`${label}: wind against Manning friction reaches q = q_inf tanh(tau t / q_inf)`, () => {
        // dq/dt = tau - g n² q² / h^(7/3): q_inf = sqrt(tau h^(7/3) / (g n²)). The update adds the stress
        // and then applies the implicit friction factor, which is first order in dt.
        const n = 0.02, h = 1.25, wind = [15, 0];
        const { config, initial } = scenario('wind-uniform', 5, 3);
        initial.manning.fill(n);
        const solver = new Solver({ ...config, wind }, initial);
        const tau = windStress(solver.config)[0], qInf = Math.sqrt(tau * h ** (7 / 3) / (config.gravity * n * n));
        for (const until of [50, 200, 2000, 8000]) {
            while (solver.time < until) solver.step(Math.min(0.05, until - solver.time));
            const exact = qInf * Math.tanh(tau * solver.time / qInf);
            assert.ok(Math.abs(solver.state[1] - exact) < 2e-3 * qInf, `t = ${until}: ${solver.state[1]} vs ${exact}`);
        }
        assert.ok(Math.abs(solver.state[1] - qInf) < 1e-3 * qInf, 'steady discharge balances wind and friction');
    });

    test(`${label}: the exact wind setup of a closed basin stays close to equilibrium`, () => {
        // Steady state without flow: g h dh/dx = tau, so h² = A + 2 tau x / g with A fixed by the volume.
        // Started from its cell averages, the solution keeps within a small fraction of the setup; the
        // deviation (a seiche excited at the walls, where the wall traces are first order) falls with dx.
        const deviation = width => {
            const L = 100, dx = L / width, h0 = 0.5, wind = [30, 0];
            const tau = windStress(configuration({ width: 1, height: 1, wind }))[0], s = 2 * tau / 9.81;
            const volume = A => 2 / (3 * s) * ((A + s * L) ** 1.5 - A ** 1.5);
            let lo = 0, hi = 1;
            for (let k = 0; k < 200; k++) { const m = (lo + hi) / 2; if (volume(m) < h0 * L) lo = m; else hi = m; }
            const A = (lo + hi) / 2, average = (x0, x1) => 2 / (3 * s * (x1 - x0)) * ((A + s * x1) ** 1.5 - (A + s * x0) ** 1.5);
            const state = new Float64Array(4 * width), bed = new Float64Array(width);
            for (let x = 0; x < width; x++) state[4 * x] = average(x * dx, (x + 1) * dx);
            const solver = new Solver({ width, height: 1, dx, dy: dx, wind }, { state, bed });
            while (solver.time < 300) solver.step(300 - solver.time);
            let error = 0;
            for (let x = 0; x < width; x++) error = Math.max(error, Math.abs(solver.state[4 * x] - average(x * dx, (x + 1) * dx)));
            const setup = Math.sqrt(A + s * L) - Math.sqrt(A);
            const volume0 = diagnostics(state, solver.config).volume;
            assert.ok(Math.abs(diagnostics(solver.state, solver.config).volume - volume0) < 1e-12 * volume0, 'volume');
            return error / setup;
        };
        const coarse = deviation(25), fine = deviation(100);
        assert.ok(fine < (label === 'MUSCL' ? 1e-3 : 5e-3), `relative deviation ${fine}`);
        assert.ok(fine < 0.7 * coarse, `deviation must fall with dx: ${coarse} -> ${fine}`);
    });

    test(`${label}: wind over a wetting and drying basin keeps depths nonnegative and closes the budget`, () => {
        const { config, initial } = scenario('wind-basin', 17, 13), solver = new Solver(config, initial);
        const volume0 = diagnostics(initial.state, config).volume;
        run(solver, 600);
        const stats = diagnostics(solver.state, solver.config);
        assert.ok(stats.minDepth >= 0 && stats.maxSpeed > 0);
        assert.ok(Math.abs(stats.volume - volume0) <= 1e-12 * volume0);
        for (let i = 0; i < initial.solid.length; i++) if (initial.solid[i]) assert.equal(solver.state[4 * i], 0);
    });
}
