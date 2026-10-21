import test from 'node:test';
import assert from 'node:assert/strict';
import { MusclReferenceSolver, hllc } from '../src/muscl-reference.mjs';
import { ReferenceSolver } from '../src/reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { diagnostics } from '../src/model.mjs';
import { exactScenario, cellAverages, depthErrors, diagonalScenario, diagonalAverages, steadyBedScenario } from '../src/analytical.mjs';

test('HLLC is consistent and resolves a moving tangential contact', () => {
    assert.deepEqual(hllc([2, 3, 4], [2, 3, 4], 9.81), [6, 37.620000000000005, 24]);
    const flux = hllc([1, 1, 2], [1, 1, -2], 1);
    assert.deepEqual(flux, [1, 1.5, 2]);
    assert.deepEqual(hllc([0, 0, 0], [0, 0, 0], 1), [0, 0, 0]);
    assert.deepEqual(hllc([1, 8, 2], [1, 9, -2], 1), [8, 64.5, 16]);
    assert.deepEqual(hllc([1, -9, 2], [1, -8, -2], 1), [-8, 64.5, 16]);
});

test('dry and separating-vacuum states use finite fallback fluxes', () => {
    let count = 0;
    for (const [a, b] of [[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 0, 0]], [[1, -3, 0], [1, 3, 0]]]) {
        const f = hllc(a, b, 1, () => count++);
        assert.ok(f.every(Number.isFinite));
    }
    assert.equal(count, 3);
});

for (const name of ['constant', 'lake', 'lake-rounded', 'shoreline', 'rain', 'dam', 'dry-dam', 'pulse']) {
    test(`MUSCL ${name}: equilibrium/budget and wet-dry regression`, () => {
        const { config, initial } = scenario(name, 17, 13), solver = new MusclReferenceSolver(config, initial);
        const volume = diagnostics(initial.state, solver.config).volume;
        for (let i = 0; i < 250; i++) solver.step(0.05);
        const stats = diagnostics(solver.state, solver.config);
        const expected = volume + solver.config.rain * solver.time * 17 * 13 * config.dx * config.dy;
        assert.ok(Math.abs(stats.volume - expected) < 1e-10 * Math.max(expected, 1));
        if (['constant', 'lake', 'shoreline', 'lake-rounded'].includes(name)) {
            const error = Math.max(...solver.state.map((v, i) => Math.abs(v - initial.state[i])));
            assert.ok(error < 1e-5, `max error ${error}`);
        }
    });
}

function solve(Solver, fixture) {
    const solver = new Solver(fixture.config, fixture.initial);
    while (fixture.finalTime - solver.time > 1e-13) solver.step(fixture.finalTime - solver.time);
    return depthErrors(solver.state, cellAverages(fixture.config.width, fixture.config.dx, fixture.sample, solver.time));
}

test('smooth nonlinear simple wave converges at second order', () => {
    const errors = [32, 64, 128, 256].map(n => solve(MusclReferenceSolver, exactScenario('simple-wave', n)).l1);
    const orders = errors.slice(1).map((v, i) => Math.log2(errors[i] / v));
    console.log('MUSCL smooth-wave L1 errors/orders:', errors, orders);
    assert.ok(orders.at(-1) >= 1.8, `finest order ${orders.at(-1)}`);
    assert.ok(errors.at(-1) < 0.25 * solve(ReferenceSolver, exactScenario('simple-wave', 256)).l1);
});

for (const name of ['wet-dam', 'dry-dam']) {
    test(`${name}: exact Riemann depth error improves over first order`, () => {
        const fixture = exactScenario(name, 256);
        const first = solve(ReferenceSolver, fixture), muscl = solve(MusclReferenceSolver, fixture);
        console.log(`${name} L1 first-order/MUSCL:`, first.l1, muscl.l1);
        assert.ok(muscl.l1 < first.l1 * 0.8);
    });
}

test('genuinely two-dimensional diagonal wave converges under refinement', () => {
    const errors = [16, 32, 64].map(n => {
        const f = diagonalScenario(n), solver = new MusclReferenceSolver(f.config, f.initial);
        while (f.finalTime - solver.time > 1e-13) solver.step(f.finalTime - solver.time);
        return depthErrors(solver.state, diagonalAverages(n, solver.time)).l1;
    });
    const order = Math.log2(errors[1] / errors[2]);
    console.log('Diagonal wave errors/order:', errors, order);
    assert.ok(order >= 1.8);
});

test('smooth moving equilibrium over a variable bed converges under refinement', () => {
    const errors = [32, 64, 128, 256].map(n => solve(MusclReferenceSolver, steadyBedScenario(n)).l1);
    const orders = errors.slice(1).map((v, i) => Math.log2(errors[i] / v));
    console.log('Moving equilibrium errors/orders:', errors, orders);
    assert.ok(orders.at(-1) >= 1.8);
});
