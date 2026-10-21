import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { windStress, configuration } from '../src/model.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];

for (const [label, Solver] of methods) {
    test(`${label}: setParameters changes wind and Coriolis between steps exactly`, () => {
        // Uniform periodic state: q = tau1 t1 + tau2 (t - t1) after the wind changes at t1.
        const { config, initial } = scenario('wind-uniform', 7, 5), solver = new Solver(config, initial);
        for (let k = 0; k < 200; k++) solver.step(0.05);
        const t1 = solver.time, tau1 = windStress(solver.config);
        solver.setParameters({ wind: [-5, 12], windDrag: 1.5e-3 });
        const tau2 = windStress(solver.config);
        for (let k = 0; k < 200; k++) solver.step(0.05);
        for (const c of [0, 1]) {
            const exact = tau1[c] * t1 + tau2[c] * (solver.time - t1);
            assert.ok(Math.abs(solver.state[1 + c] - exact) < 1e-12, `${solver.state[1 + c]} vs ${exact}`);
        }
        // Coriolis switched on mid-run: the current then turns at the new rate from its value at that time.
        const q0 = [solver.state[1], solver.state[2]], t2 = solver.time;
        solver.setParameters({ wind: [0, 0], coriolis: 1e-3 });
        for (let k = 0; k < 200; k++) solver.step(0.05);
        const ft = 1e-3 * (solver.time - t2);
        assert.ok(Math.abs(solver.state[1] - (q0[0] * Math.cos(ft) + q0[1] * Math.sin(ft))) < 1e-12);
        assert.ok(Math.abs(solver.state[2] - (-q0[0] * Math.sin(ft) + q0[1] * Math.cos(ft))) < 1e-12);
    });

    test(`${label}: setHydrographs mid-run equals one run with the combined series`, () => {
        const { config, initial } = scenario('hydrograph-channel', 17, 9);
        const original = config.hydrographs, t1 = 6;
        // Discharge raised from 2 to 3.5 m3/s from t1 on, as a series on the solver clock.
        const changed = [{ ...original[0], series: [[0, 0], [t1, 2 * t1 / 20], [t1 + 4, 3.5]] }, original[1]];
        const a = new Solver(config, initial);
        while (a.time < t1) a.step(Math.min(0.05, t1 - a.time));
        a.setHydrographs(changed);
        while (a.time < 15) a.step(Math.min(0.05, 15 - a.time));
        const b = new Solver({ ...config, hydrographs: changed }, initial);
        while (b.time < t1) b.step(Math.min(0.05, t1 - b.time));
        while (b.time < 15) b.step(Math.min(0.05, 15 - b.time));
        // Before t1 the two series agree, so the runs are identical throughout.
        for (let k = 0; k < a.state.length; k++) assert.ok(Math.abs(a.state[k] - b.state[k]) < 1e-12, `component ${k}`);
        assert.ok(Math.abs(a.boundaryOutflow - b.boundaryOutflow) < 1e-9);
    });

    test(`${label}: run-time changes are validated`, () => {
        const { config, initial } = scenario('hydrograph-channel', 9, 5), solver = new Solver(config, initial);
        assert.throws(() => solver.setParameters({ dx: 2 }), RangeError);
        assert.throws(() => solver.setParameters({ wind: [500, 0] }), RangeError);
        assert.throws(() => solver.setParameters({ settling: { velocity: 1e-3 } }), RangeError, 'needs a tracer');
        assert.throws(() => solver.setHydrographs([config.hydrographs[0]]), RangeError, 'segments must stay');
        assert.throws(() => solver.setHydrographs([{ ...config.hydrographs[0], kind: 'stage' }, config.hydrographs[1]]), RangeError);
        assert.doesNotThrow(() => configuration(config));
    });
}
