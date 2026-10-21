import test from 'node:test';
import assert from 'node:assert/strict';
import { ReferenceSolver } from '../src/reference.mjs';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { thackerScenario, thackerAverages, bumpScenario, bumpAverages, bumpSolution, thackerPeriod } from '../src/benchmarks.mjs';
import { diagnostics } from '../src/model.mjs';

const runTo = (solver, t) => { while (t - solver.time > 1e-12 * Math.max(1, t)) solver.step(t - solver.time); return solver; };
const meanDepthError = (state, exact, stride) => {
    let e = 0;
    for (let i = 0; i < exact.length / stride; i++) e += Math.abs(state[4 * i] - exact[stride * i]);
    return e / (exact.length / stride);
};

test('exact SWASHES solutions match published values', () => {
    assert.ok(Math.abs(bumpSolution('shock').jump - 11.6656) < 1e-3, 'hydraulic jump position');
    assert.ok(Math.abs(thackerPeriod() - 4.4857) < 1e-4, 'Thacker period');
    const { state } = thackerAverages(100, 0);
    let volume = 0;
    for (let k = 0; k < state.length; k += 4) volume += state[k] * 0.04 * 0.04;
    assert.ok(Math.abs(volume - Math.PI * 0.1 / 2) < 1e-5, `Thacker volume ${volume}`); // pi a² h0 / 2
});

// Coarse-grid regression gates (about twice the measured errors); MUSCL must beat Rusanov.
test('Thacker paraboloid, 32² cells, one period: MUSCL error is small, below Rusanov, volume exact', () => {
    const errors = [ReferenceSolver, MusclReferenceSolver].map(Solver => {
        const { config, initial, period } = thackerScenario(32), solver = runTo(new Solver(config, initial), period);
        const volume0 = diagnostics(initial.state, solver.config).volume, volume = diagnostics(solver.state, solver.config).volume;
        assert.ok(Math.abs(volume - volume0) < 1e-12 * volume0, 'closed basin conserves volume');
        return meanDepthError(solver.state, thackerAverages(32, solver.time).state, 4);
    });
    assert.ok(errors[1] < 6e-3 && errors[1] < errors[0], `L1 Rusanov ${errors[0]}, MUSCL ${errors[1]}`);
});

for (const [kind, gate] of [['subcritical', 3e-3], ['transcritical', 8e-3], ['shock', 7e-3]]) {
    test(`bump ${kind}, 50 cells, 200 s: MUSCL reaches the exact steady state, below Rusanov`, () => {
        const errors = [ReferenceSolver, MusclReferenceSolver].map(Solver => {
            const { config, initial, finalTime } = bumpScenario(kind, 50), solver = runTo(new Solver(config, initial), finalTime);
            return meanDepthError(solver.state, bumpAverages(kind, 50).h, 1);
        });
        assert.ok(errors[1] < gate && errors[1] < errors[0], `L1 Rusanov ${errors[0]}, MUSCL ${errors[1]}`);
    });
}
