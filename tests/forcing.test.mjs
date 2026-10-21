import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario, frictionExact } from '../src/scenarios.mjs';
import { diagnostics } from '../src/model.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];
const run = (solver, steps, maxDt = 0.05) => { for (let i = 0; i < steps; i++) solver.step(maxDt); return solver; };
const budgetResidual = (solver, volume0) =>
    diagnostics(solver.state, solver.config).volume + solver.boundaryOutflow - volume0 - solver.sourceVolume;

for (const [label, Solver] of methods) {
    test(`${label}: uniform Manning decay matches the exact local solution`, () => {
        const { config, initial } = scenario('friction', 9, 7), solver = run(new Solver(config, initial), 400);
        const exact = frictionExact(Array.from(initial.state.subarray(0, 3)), Math.fround(0.03), config.gravity, solver.time);
        for (let k = 0; k < solver.state.length; k += 4) for (let c = 0; c < 3; c++) {
            assert.ok(Math.abs(solver.state[k + c] - exact[c]) < 1e-9 * Math.max(1, Math.abs(exact[c])),
                `component ${c}: ${solver.state[k + c]} vs ${exact[c]}`);
        }
        assert.ok(Math.hypot(solver.state[1], solver.state[2]) < 0.8 * Math.hypot(0.4, 0.3), 'friction must act');
    });

    test(`${label}: spatial sources and rain close the water budget from a dry start`, () => {
        const { config, initial } = scenario('sources', 17, 13), solver = run(new Solver(config, initial), 400);
        const stats = diagnostics(solver.state, solver.config);
        assert.ok(stats.minDepth >= 0 && solver.sourceVolume > 0);
        assert.ok(Math.abs(budgetResidual(solver, 0)) <= 1e-10 * solver.sourceVolume);
    });

    test(`${label}: solid cells stay dry and impermeable; closed budget`, () => {
        const { config, initial } = scenario('obstacle', 17, 13), solver = new Solver(config, initial);
        const volume0 = diagnostics(initial.state, solver.config).volume;
        run(solver, 400);
        for (let i = 0; i < initial.solid.length; i++) if (initial.solid[i]) {
            assert.deepEqual(Array.from(solver.state.subarray(4 * i, 4 * i + 3)), [0, 0, 0]);
        }
        assert.ok(Math.abs(budgetResidual(solver, volume0)) < 1e-11 * volume0);
        assert.equal(solver.boundaryOutflow, 0);
    });

    test(`${label}: lake at rest with obstacles and shoreline is preserved`, () => {
        const { config, initial } = scenario('lake-obstacle', 17, 13), solver = run(new Solver(config, initial), 500);
        const error = Math.max(...solver.state.map((v, i) => Math.abs(v - initial.state[i])));
        assert.ok(error < 1e-12, `max state error ${error}`);
    });

    test(`${label}: open boundaries keep a uniform flow unchanged`, () => {
        const { config, initial } = scenario('open-uniform', 9, 11), solver = run(new Solver(config, initial), 300);
        const error = Math.max(...solver.state.map((v, i) => Math.abs(v - initial.state[i])));
        assert.ok(error < 1e-12, `max state error ${error}`);
        // Net outflow equals inflow for uniform flow through opposite open faces.
        assert.ok(Math.abs(solver.boundaryOutflow) < 1e-9);
    });

    test(`${label}: open-boundary outflow closes the budget as water leaves`, () => {
        const { config, initial } = scenario('open-dam', 21, 17), solver = new Solver(config, initial);
        const volume0 = diagnostics(initial.state, solver.config).volume;
        run(solver, 400);
        assert.ok(solver.boundaryOutflow > 0.05 * volume0, `outflow ${solver.boundaryOutflow}`);
        assert.ok(Math.abs(budgetResidual(solver, volume0)) < 1e-11 * volume0);
        assert.ok(diagnostics(solver.state, solver.config).minDepth >= 0);
    });

    test(`${label}: outflow boundaries let water leave but never enter`, () => {
        const { config, initial } = scenario('outflow-drain', 9, 11), solver = new Solver(config, initial);
        const volume0 = diagnostics(initial.state, solver.config).volume;
        let previous = 0;
        for (let i = 0; i < 300; i++) {
            solver.step(0.05);
            assert.ok(solver.boundaryOutflow >= previous - 1e-12, `outflow decreased at step ${i}`);
            previous = solver.boundaryOutflow;
        }
        assert.ok(solver.boundaryOutflow > 0.05 * volume0);
        assert.ok(Math.abs(budgetResidual(solver, volume0)) < 1e-11 * volume0);
        assert.ok(diagnostics(solver.state, solver.config).minDepth >= 0);
        const lake = scenario('lake', 9, 11);
        const still = run(new Solver({ ...lake.config, boundaryX: 'outflow', boundaryY: 'outflow' }, lake.initial), 300);
        assert.ok(Math.max(...still.state.map((v, i) => Math.abs(v - lake.initial.state[i]))) < 1e-12);
    });

    test(`${label}: forcing can change between steps; solid mask cannot`, () => {
        const { config, initial } = scenario('sources', 9, 9), solver = new Solver(config, initial);
        run(solver, 20);
        const before = solver.sourceVolume;
        solver.setForcing({ source: new Float64Array(81), manning: new Float64Array(81).fill(0.05) });
        const dt = solver.step(0.05);
        assert.ok(Math.abs(solver.sourceVolume - before - config.rain * 81 * config.dx * config.dy * dt) < 1e-15);
        assert.throws(() => solver.setForcing({ source: new Float64Array(81).fill(-1) }));
        assert.throws(() => new Solver(config, { ...initial, solid: new Uint8Array(81).fill(1),
            state: new Float32Array(4 * 81).fill(1) }));
    });
}

test('MUSCL and Rusanov agree on the transposed obstacle problem', () => {
    const fixture = scenario('obstacle', 13, 9);
    const { width: w, height: h } = fixture.config;
    const t = { state: new Float32Array(4 * w * h), bed: new Float32Array(w * h), solid: new Uint8Array(w * h) };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, j = x * h + y;
        t.state.set([fixture.initial.state[4 * i], fixture.initial.state[4 * i + 2], fixture.initial.state[4 * i + 1], 0], 4 * j);
        t.solid[j] = fixture.initial.solid[i];
    }
    for (const Solver of [ReferenceSolver, MusclReferenceSolver]) {
        const a = run(new Solver(fixture.config, fixture.initial), 100);
        const b = run(new Solver({ ...fixture.config, width: h, height: w, dx: fixture.config.dy, dy: fixture.config.dx }, t), 100);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const i = 4 * (y * w + x), j = 4 * (x * h + y);
            assert.ok(Math.abs(a.state[i] - b.state[j]) < 1e-12);
            assert.ok(Math.abs(a.state[i + 1] - b.state[j + 2]) < 1e-12);
        }
    }
});
