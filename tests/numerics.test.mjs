import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { diagnostics } from '../src/model.mjs';
import { scenario } from '../src/scenarios.mjs';

function make(name, w = 17, h = 33) {
    const { config, initial } = scenario(name, w, h);
    return new ReferenceSolver(config, initial);
}
function advance(solver, steps = 1000, maxDt = 0.05) {
    for (let i = 0; i < steps; i++) solver.step(maxDt);
    return diagnostics(solver.state, solver.config);
}

for (const name of ['constant', 'lake', 'shoreline']) {
    test(`${name}: preserve analytical equilibrium for 1000 steps`, () => {
        const solver = make(name), initial = solver.readState();
        advance(solver);
        const error = Math.max(...solver.state.map((v, i) => Math.abs(v - initial[i])));
        assert.ok(error < 1e-12, `max state error ${error}`);
    });
}

for (const name of ['dam', 'dry-dam', 'pulse']) {
    test(`${name}: conserve closed-domain volume with nonnegative finite state`, () => {
        const solver = make(name);
        const volume = diagnostics(solver.state, solver.config).volume;
        const result = advance(solver);
        assert.ok(Math.abs(result.volume - volume) / volume < 1e-12);
        assert.ok(result.minDepth >= 0);
        assert.ok(solver.time > 0);
    });
}

test('rain accumulates on initially dry cells below the old 1 mm threshold', () => {
    const solver = make('rain');
    const result = advance(solver, 1000, 0.05);
    const expected = solver.config.rain * solver.time;
    assert.ok(expected < 0.001);
    for (let k = 0; k < solver.state.length; k += 4) {
        assert.ok(Math.abs(solver.state[k] - expected) < 1e-15);
        assert.equal(solver.state[k + 1], 0);
        assert.equal(solver.state[k + 2], 0);
    }
    assert.ok(Math.abs(result.volume - expected * 17 * 33 * 1.5) < 1e-12);
});

test('lake on rounded elevations stays within the equilibrium tolerance', () => {
    const solver = make('lake-rounded'), initial = solver.readState();
    const result = advance(solver);
    const error = Math.max(...solver.state.map((v, i) => Math.abs(v - initial[i])));
    assert.ok(error < 1e-5, `max state error ${error}`);
    assert.ok(result.maxSpeed < 1e-5 * Math.sqrt(9.81 * 2));
});

test('a rejected nonfinite update preserves accepted state and time', () => {
    const { config, initial } = scenario('rain', 1, 1);
    const solver = new ReferenceSolver({ ...config, rain: Number.MAX_VALUE }, initial);
    const before = solver.readState();
    assert.throws(() => solver.step(2), /Rejected invalid update|Invalid timestep/);
    assert.deepEqual(solver.state, before);
    assert.equal(solver.time, 0);
});

test('global CFL includes both directions and advection; clock uses accepted dt', () => {
    const { config, initial } = scenario('constant', 1, 1);
    initial.state.set([1, 3, 4, 0]);
    const solver = new ReferenceSolver(config, initial);
    const expected = config.cfl / ((3 + 2 * Math.sqrt(config.gravity)) / config.dx
        + (4 + 2 * Math.sqrt(config.gravity)) / config.dy);
    assert.equal(solver.step(10), expected);
    assert.equal(solver.time, expected);
    assert.deepEqual(solver.state, new Float64Array([1, 3, 4, 0]));
});

test('one-cell wall domain has no mass loss', () => {
    const solver = make('lake', 1, 1), volume = diagnostics(solver.state, solver.config).volume;
    assert.equal(advance(solver).volume, volume);
});

test('x/y transposition preserves dynamics on rectangular cells', () => {
    const fixture = scenario('dam', 7, 19);
    const transposed = { state: new Float32Array(fixture.initial.state.length), bed: new Float32Array(7 * 19) };
    for (let y = 0; y < 19; y++) for (let x = 0; x < 7; x++) {
        const i = 4 * (y * 7 + x), j = 4 * (x * 19 + y);
        transposed.state.set([fixture.initial.state[i], fixture.initial.state[i + 2], fixture.initial.state[i + 1], 0], j);
    }
    const a = new ReferenceSolver(fixture.config, fixture.initial);
    const b = new ReferenceSolver({ ...fixture.config, width: 19, height: 7, dx: 1.5, dy: 1 }, transposed);
    advance(a, 100); advance(b, 100);
    for (let y = 0; y < 19; y++) for (let x = 0; x < 7; x++) {
        const i = 4 * (y * 7 + x), j = 4 * (x * 19 + y);
        assert.ok(Math.abs(a.state[i] - b.state[j]) < 1e-12);
        assert.ok(Math.abs(a.state[i + 1] - b.state[j + 2]) < 1e-12);
    }
});

test('invalid inputs and unsupported boundaries fail explicitly', () => {
    const { config, initial } = scenario('lake', 2, 2);
    assert.throws(() => new ReferenceSolver({ ...config, boundaryX: 'sponge' }, initial));
    assert.throws(() => new ReferenceSolver({ ...config, cfl: 0.8 }, initial));
    assert.throws(() => new ReferenceSolver({ ...config, dx: 0 }, initial));
    assert.throws(() => new ReferenceSolver({ ...config, rain: -1 }, initial));
    assert.throws(() => new ReferenceSolver(config, { ...initial, state: new Float32Array(1) }));
    initial.state[0] = -1;
    assert.throws(() => new ReferenceSolver(config, initial));
    const solver = make('lake');
    for (const dt of [0, -1, NaN, Infinity]) assert.throws(() => solver.step(dt));
    assert.equal(solver.time, 0);
});

test('input arrays and returned snapshots cannot mutate solver state', () => {
    const { config, initial } = scenario('lake', 2, 2);
    const solver = new ReferenceSolver(config, initial), before = solver.readState();
    initial.state.fill(0); initial.bed.fill(100);
    solver.readState().fill(0);
    assert.deepEqual(solver.state, before);
    solver.step(0.01);
    assert.deepEqual(solver.state, before);
});
