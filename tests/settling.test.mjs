import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { configuration, settledConcentration, tracerBounds, initialData } from '../src/model.mjs';

const methods = [['Rusanov', ReferenceSolver], ['MUSCL', MusclReferenceSolver]];
const sum = a => a.reduce((s, v) => s + v, 0);

test('settling options are validated', () => {
    const base = { width: 2, height: 2, tracer: true };
    assert.throws(() => configuration({ width: 2, height: 2, settling: { velocity: 1e-3 } }), RangeError, 'requires a tracer');
    for (const settling of [{}, { velocity: -1 }, { velocity: 2 }, { velocity: NaN }, { velocity: 1e-3, exponent: -0.5 }, { velocity: 1e-3, exponent: 4 }]) {
        assert.throws(() => configuration({ ...base, settling }), RangeError, JSON.stringify(settling));
    }
    assert.deepEqual(configuration({ ...base, settling: { velocity: 1e-3 } }).settling, { velocity: 1e-3, exponent: 0 });
});

test('the local settling update is the exact solution and composes over steps', () => {
    // Constant: c exp(-w t / h). Flocculating (w = k c^m): c^-m grows linearly, c = (c0^-m + m k t / h)^(-1/m).
    const constant = { velocity: 2e-3, exponent: 0 }, floc = { velocity: 1.5e-3, exponent: 4 / 3 };
    assert.ok(Math.abs(settledConcentration(constant, 0.8, 1.5, 300) - 0.8 * Math.exp(-2e-3 * 300 / 1.5)) < 1e-15);
    const exact = (0.8 ** (-4 / 3) + (4 / 3) * 1.5e-3 * 300 / 1.5) ** (-3 / 4);
    assert.ok(Math.abs(settledConcentration(floc, 0.8, 1.5, 300) - exact) < 1e-14);
    for (const s of [constant, floc]) {
        let c = 0.8;
        for (let k = 0; k < 600; k++) c = settledConcentration(s, c, 1.5, 0.5);
        assert.ok(Math.abs(c - settledConcentration(s, 0.8, 1.5, 300)) < 1e-13, 'steps compose to the exact solution');
        assert.equal(settledConcentration(s, 0, 1.5, 10), 0);
        assert.equal(settledConcentration(s, 0.5, 0, 10), 0.5, 'dry cells are left to the dry-cell rule');
    }
    // A flocculating suspension settles faster when more concentrated (relative loss over one step).
    const loss = c => 1 - settledConcentration(floc, c, 1, 10) / c;
    assert.ok(loss(2) > loss(1) && loss(1) > loss(0.1));
});

test('settling widens the concentration bounds to zero', () => {
    const { config, initial } = scenario('lake', 7, 5);
    const n = config.width * config.height, concentration = new Float64Array(n).fill(0.6);
    const p = configuration({ ...config, tracer: true, settling: { velocity: 1e-3 } });
    assert.deepEqual(tracerBounds(p, initialData(p, { ...initial, concentration })), [0, 0.6]);
});

for (const [label, Solver] of methods) {
    test(`${label}: a lake at rest settles cell by cell exactly`, () => {
        for (const settling of [{ velocity: 2e-3, exponent: 0 }, { velocity: 1.5e-3, exponent: 4 / 3 }]) {
            const { config, initial } = scenario('lake', 9, 7);
            const n = config.width * config.height, concentration = new Float64Array(n).fill(0.8);
            const solver = new Solver({ ...config, tracer: true, settling }, { ...initial, concentration });
            const amount0 = sum(Array.from(solver.tracer)) * config.dx * config.dy;
            while (solver.time < 200) solver.step(200 - solver.time);
            for (let i = 0; i < n; i++) {
                const h = solver.state[4 * i];
                if (!(h > 0)) continue;
                const exact = settledConcentration(solver.config.settling, 0.8, h, solver.time);
                assert.ok(Math.abs(solver.tracer[i] / h - exact) < 1e-12, `cell ${i}: ${solver.tracer[i] / h} vs ${exact}`);
            }
            const remaining = sum(Array.from(solver.tracer)) * config.dx * config.dy;
            assert.ok(Math.abs(remaining + solver.tracerDeposited - amount0) < 1e-12 * amount0, 'deposit closes the budget');
            assert.ok(solver.tracerDeposited > 0.01 * amount0, 'settling must act');
        }
    });

    test(`${label}: settling in moving water keeps concentrations in range and closes the tracer budget`, () => {
        for (const [name, settling] of [['dam', { velocity: 2e-3 }], ['sources', { velocity: 1e-3, exponent: 4 / 3 }],
            ['hydrograph-channel', { velocity: 1.5e-3, exponent: 1 }]]) {
            const { config, initial } = scenario(name, 17, 13);
            const n = config.width * config.height, concentration = new Float64Array(n), sourceConcentration = new Float64Array(n).fill(1.5);
            for (let i = 0; i < n; i++) concentration[i] = ((i * 7) % 11) / 10;
            const hydrographs = (config.hydrographs ?? []).map(h => ({ ...h, concentration: 2 }));
            const solver = new Solver({ ...config, tracer: true, settling, hydrographs }, { ...initial, concentration, sourceConcentration });
            const amount0 = sum(Array.from(solver.tracer)) * config.dx * config.dy;
            for (let k = 0; k < 400; k++) solver.step(0.05);
            const [lo, hi] = solver.bounds;
            assert.equal(lo, 0);
            for (let i = 0; i < n; i++) {
                const h = solver.state[4 * i];
                if (h > 1e-4) { const c = solver.tracer[i] / h; assert.ok(c >= -1e-12 && c <= hi + 1e-12, `${name} cell ${i}: c = ${c}`); }
            }
            const remaining = sum(Array.from(solver.tracer)) * config.dx * config.dy;
            const residual = remaining + solver.tracerOutflow + solver.tracerDeposited - amount0 - solver.tracerSourceAmount;
            assert.ok(Math.abs(residual) <= 1e-11 * Math.max(amount0 + solver.tracerSourceAmount, 1), `${name}: residual ${residual}`);
            assert.ok(solver.tracerDeposited > 0, `${name}: settling must act`);
        }
    });
}
