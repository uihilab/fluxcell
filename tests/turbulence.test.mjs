import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { scenario } from '../src/scenarios.mjs';
import { viscousShearCase, turbulenceCase, meanAbsoluteError } from '../src/transport.mjs';

const run = (solver, until) => { while (solver.time < until - 1e-12) solver.step(until - solver.time); return solver; };
const sum = (a, k) => { let s = 0; for (let i = k; i < a.length; i += 4) s += a[i]; return s; };

test('a constant eddy viscosity reproduces the viscous shear layer at second order', () => {
    for (const Solver of [ReferenceSolver, MusclReferenceSolver]) {
        const errors = [];
        for (const n of [32, 64, 128]) {
            const { config, initial, exact } = viscousShearCase(n);
            const s = run(new Solver(config, initial), 20);
            errors.push(meanAbsoluteError(Array.from({ length: s.state.length / 4 }, (_, i) => s.state[4 * i + 1] / s.state[4 * i]), exact(s.time).velocity));
        }
        const order = Math.log2(errors[1] / errors[2]);
        if (Solver === MusclReferenceSolver) assert.ok(order > 1.7, `MUSCL order ${order} (${errors})`);
        else assert.ok(errors[2] < errors[0], `Rusanov errors ${errors}`);
    }
});

test('the closure leaves a lake at rest and uniform flow unchanged', () => {
    for (const name of ['lake', 'constant']) {
        const { config, initial } = scenario(name, 17, 13);
        const s = new MusclReferenceSolver({ ...config, smagorinsky: 0.2, bedTurbulence: 0.067, eddyViscosity: 0.01 }, { ...initial, manning: new Float64Array(17 * 13).fill(0) });
        for (let i = 0; i < 100; i++) s.step(1);
        let d = 0;
        for (let k = 0; k < s.state.length; k++) if (k % 4 !== 3) d = Math.max(d, Math.abs(s.state[k] - initial.state[k]));
        assert.ok(d < 1e-12, `${name} changed by ${d}`);
    }
});

test('Smagorinsky viscosity conserves mass and momentum in a periodic domain and removes energy', () => {
    const { config, initial } = turbulenceCase(32);
    const s = new MusclReferenceSolver({ ...config, smagorinsky: 0.2 }, initial), base = new MusclReferenceSolver(config, initial);
    const m0 = sum(initial.state, 0), px = sum(initial.state, 1), py = sum(initial.state, 2);
    run(s, 30); run(base, 30);
    assert.ok(Math.abs(sum(s.state, 0) - m0) < 1e-11 * m0, 'mass');
    assert.ok(Math.abs(sum(s.state, 1) - px) < 1e-9 && Math.abs(sum(s.state, 2) - py) < 1e-9, `momentum ${sum(s.state, 1) - px} ${sum(s.state, 2) - py}`);
    const energy = st => { let e = 0; for (let k = 0; k < st.length; k += 4) e += (st[k + 1] ** 2 + st[k + 2] ** 2) / st[k]; return e; };
    assert.ok(energy(s.state) < energy(base.state), 'the closure dissipates energy');
});

test('turbulent tracer diffusivity follows nu / schmidt', () => {
    const { config, initial } = viscousShearCase(32);
    const c = Float64Array.from({ length: initial.bed.length }, (_, i) => (Math.floor(i / 4) < 16 ? 1 : 0));
    const withNu = run(new MusclReferenceSolver({ ...config, tracer: true, schmidt: 0.5 }, { ...initial, concentration: c }), 5);
    const plain = run(new MusclReferenceSolver({ ...config, eddyViscosity: 0, tracer: true, diffusivity: 0.05 / 0.5 }, { ...initial, concentration: c }), 5);
    // Same diffusivity (0.1 m²/s) by both routes; the momentum differs, so compare the tracer only.
    const diff = Math.max(...Array.from(withNu.tracer, (v, i) => Math.abs(v - plain.tracer[i])));
    assert.ok(diff < 1e-3, `tracer difference ${diff}`);
});

test('closure options are validated', () => {
    const base = { width: 4, height: 3 }, initial = { state: new Float64Array(48), bed: new Float64Array(12) };
    for (const bad of [{ eddyViscosity: -1 }, { smagorinsky: 0.5 }, { bedTurbulence: 2 }, { schmidt: 0 }]) {
        assert.throws(() => new ReferenceSolver({ ...base, ...bad }, initial), RangeError);
    }
});
