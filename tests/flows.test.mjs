import test from 'node:test';
import assert from 'node:assert/strict';
import { MusclReferenceSolver } from '../src/muscl-reference.mjs';
import { conjugateDepth, obliqueShock, hydraulicJumpCase, obliqueShockCase, besselJ, lambDipole, dipoleCase, vorticityCentroid } from '../src/flows.mjs';

test('reference relations: Belanger, Ippen and the Bessel functions', () => {
    assert.ok(Math.abs(conjugateDepth(0.5, 3) - 0.25 * (Math.sqrt(73) - 1)) < 1e-15);
    // Ippen: at Fr = 3, theta = 10 degrees the weak wave lies at about 28-29 degrees (above the Mach angle 19.5).
    const { beta, ratio } = obliqueShock(3, 10 * Math.PI / 180);
    assert.ok(beta > Math.asin(1 / 3) && beta < 35 * Math.PI / 180, `beta = ${beta * 180 / Math.PI}`);
    const t = Math.tan(beta), s = Math.sqrt(1 + 8 * 9 * Math.sin(beta) ** 2);
    assert.ok(Math.abs(Math.atan(t * (s - 3) / (2 * t * t + s - 1)) - 10 * Math.PI / 180) < 1e-10);
    assert.ok(Math.abs(ratio - (s - 1) / 2) < 1e-15);
    // J0(2.404825557695773) = 0, J1(3.8317059702075125) = 0, J0(1) = 0.7651976865579666.
    assert.ok(Math.abs(besselJ(0, 2.404825557695773)) < 1e-14 && Math.abs(besselJ(1, 3.8317059702075125)) < 1e-14);
    assert.ok(Math.abs(besselJ(0, 1) - 0.7651976865579666) < 1e-15);
    // The dipole's flow is divergence free and continuous at r = a.
    const d = { cx: 0, cy: 0, a: 8, U: 0.2 }, e = 1e-3;
    for (const [x, y] of [[3, 2], [-5, 1], [9, -4], [7.99, 0.3]]) {
        const div = (lambDipole(x + e, y, d)[0] - lambDipole(x - e, y, d)[0] + lambDipole(x, y + e, d)[1] - lambDipole(x, y - e, d)[1]) / (2 * e);
        assert.ok(Math.abs(div) < 1e-6, `divergence ${div} at ${x}, ${y}`);
    }
    const inside = lambDipole(7.999, 1, d), outside = lambDipole(8.001, 1, d);
    assert.ok(Math.hypot(inside[0] - outside[0], inside[1] - outside[1]) < 1e-3);
});

test('MUSCL: a hydraulic jump stays at its Belanger conjugate depths', () => {
    const c = hydraulicJumpCase(1, { width: 2 }), solver = new MusclReferenceSolver(c.config, c.initial);
    while (solver.time < 60) solver.step(60 - solver.time);
    const nx = c.config.width, h = Array.from({ length: nx }, (_, x) => solver.state[4 * x]);
    const jump = h.findIndex(v => v > (c.h1 + c.h2) / 2);
    const upstream = h[Math.floor(jump / 2)], downstream = h[Math.floor((jump + nx) / 2)];
    assert.ok(Math.abs(upstream - c.h1) < 0.01 * c.h1, `upstream depth ${upstream}`);
    assert.ok(Math.abs(downstream - c.h2) < 0.01 * c.h2, `downstream depth ${downstream}`);
    assert.ok(Math.abs(jump - nx / 2) <= 6, `jump at cell ${jump} of ${nx}`);
});

test("MUSCL: an oblique jump at a wall forms at Ippen's angle and depth ratio, converging with dx", () => {
    const measure = dx => {
        const c = obliqueShockCase(dx), solver = new MusclReferenceSolver(c.config, c.initial);
        while (solver.time < 25) solver.step(25 - solver.time);
        const { width: nx } = c.config, mid = c.h1 * (1 + c.ratio) / 2, points = [];
        // Shock position along columns 30 to 65 m past the leading corner (the shock bends near the
        // one-cell step at the wall's leading edge): the lowest crossing of the mid depth from above.
        for (let x = Math.round((c.x0 + 30) / dx); x < Math.round((c.x0 + 65) / dx); x++) {
            for (let y = Math.round(40 / dx); y > 1; y--) {
                const a = solver.state[4 * (y * nx + x)], b = solver.state[4 * ((y - 1) * nx + x)];
                if (a < mid && b >= mid) { points.push([(x + 0.5) * dx, (y - 0.5 + (a - mid) / (a - b)) * dx]); break; }
            }
        }
        const m = points.length, sx = points.reduce((t, p) => t + p[0], 0) / m, sy = points.reduce((t, p) => t + p[1], 0) / m;
        const slope = points.reduce((t, p) => t + (p[0] - sx) * (p[1] - sy), 0) / points.reduce((t, p) => t + (p[0] - sx) ** 2, 0);
        // Depth between the wall and the shock 50 m past the corner, a third of the way up.
        const X = c.x0 + 50, x = Math.round(X / dx), y = Math.round((sy + slope * (X - sx)) / 3 / dx);
        return { c, angle: Math.atan(slope), ratio: solver.state[4 * (y * nx + x)] / c.h1 };
    };
    const coarse = measure(1), fine = measure(0.5), c = fine.c, expected = c.beta - c.theta;
    assert.ok(Math.abs(fine.angle - expected) < 0.05 * expected, `angle ${fine.angle * 180 / Math.PI} vs Ippen ${expected * 180 / Math.PI}`);
    assert.ok(Math.abs(fine.ratio - c.ratio) < 0.02 * c.ratio, `depth ratio ${fine.ratio} vs ${c.ratio}`);
    assert.ok(Math.abs(fine.angle - expected) <= Math.abs(coarse.angle - expected) + 1e-3, 'angle converges');
});

test('MUSCL: a Lamb-Chaplygin dipole translates at its speed U', () => {
    const U = 0.2, n = 96, c = dipoleCase(n, [{ cx: 30, cy: 50, a: 10, U }], { tracer: false });
    const solver = new MusclReferenceSolver({ ...c.config, lowFroude: true }, c.initial), dx = c.config.dx;
    const start = vorticityCentroid(solver.state, n, dx, 1, [30, 50]);
    while (solver.time < 100) solver.step(100 - solver.time);
    const end = vorticityCentroid(solver.state, n, dx, 1, [30 + U * 100, 50]);
    const speed = (end[0] - start[0]) / solver.time;
    assert.ok(Math.abs(speed - U) < 0.1 * U, `speed ${speed} vs ${U}`);
    assert.ok(Math.abs(end[1] - start[1]) < 1, 'travels straight');
});
