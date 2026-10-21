// Vortices and rotation: shear-layer instability, wake shedding, 2D turbulence, exact vortices and
// dipoles, and geostrophic adjustment.
import { shearLayerCase, wakeCase, turbulenceCase, vortexCase } from '../../../src/transport.mjs';
import { dipoleCase } from '../../../src/flows.mjs';

// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // Shallow-flow dynamics (src/transport.mjs; quantitative checks in docs/VERIFICATION.md).
    'kelvin-helmholtz'({ U }) {
        return { ...shearLayerCase(384, { U, amplitude: 1e-2, width: 40 }), depthScale: 10, wetDepth: 0.01, rate: 1.5, colour: 'concentration',
            concentration: 'linear', vorticityScale: 0.6 * U, speedScale: 1.2 * U, exaggeration: 1, surfaceGain: 60, still: 10, elevationScale: 0.03, hideGround: true,
            note: `Shear layer u = U tanh(y/δ) with U = ${U} m/s, perturbed at its most unstable wavelength; the tracer marks the two streams. The billows grow at a rate proportional to U.` };
    },
    'vortex-street'({ u }) {
        const c = wakeCase(0.5, { u }), { width: W, height: H } = c.config, dx = 0.5, shape = new Float32Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
            shape[y * W + x] = (Math.hypot((x + 0.5) * dx - 80, (y + 0.5) * dx - (50 + 0.3 * dx)) - 5) / dx;
        return { ...c, shape, inflow: u, cylinders: [{ x: 80, y: 50 + 0.3 * dx, r: 5 }], cylinderTop: 3, particles: { seed: [0.1, 0.38, 0.17, 0.62], life: [4000, 3000], count: 9000 }, depthScale: 2, wetDepth: 0.01, rate: 6, colour: 'vorticity', concentration: 'linear',
            view3d: true, exaggeration: 1, surfaceGain: 80, still: 2, elevationScale: 0.01, buildingHeight: 0, hideGround: true,
            camera: { target: [-60, 0, 0], distance: 190, yaw: -1.9, pitch: 0.75 },
            vorticityScale: 0.5 * u, speedScale: 1.6 * u,
            note: `${u} m/s past a 10 m island in water 2 m deep; a dye band enters on the centreline and particles are released just upstream of the island. Vortices shed at a Strouhal number of about 0.2, every ${(10 / (0.2 * u)).toFixed(0)} s here; slower flow sheds less.` };
    },
    turbulence() {
        return { ...turbulenceCase(256), depthScale: 10, wetDepth: 0.01, rate: 1.5, colour: 'vorticity', concentration: 'linear',
            vorticityScale: 0.5, speedScale: 1, exaggeration: 1, surfaceGain: 60, still: 10, elevationScale: 0.03, hideGround: true,
            note: 'Decaying 2D turbulence: vortices merge and energy moves to larger scales while the tracer is stirred.' };
    },
    vortex() {
        return { ...vortexCase(256), depthScale: 1, wetDepth: 0.01, rate: 4, colour: 'concentration', concentration: 'linear',
            vorticityScale: 0.3, speedScale: 1, exaggeration: 1, surfaceGain: 60, still: 1, elevationScale: 0.1, hideGround: true,
            note: 'Steady vortex in cyclostrophic balance (an exact solution); the tracer front winds into a spiral.' };
    },
    // Two Lamb-Chaplygin dipoles meet head on (each translates at U: tests/flows.test.mjs).
    dipoles({ U, offset }) {
        const c = dipoleCase(256, [{ cx: 25, cy: 50 + offset / 2, a: 10, U }, { cx: 75, cy: 50 - offset / 2, a: 10, U, phi: Math.PI }]);
        return { ...c, config: { ...c.config, lowFroude: true }, depthScale: 10, wetDepth: 0.01, rate: 4, colour: 'vorticity',
            concentration: 'linear', vorticityScale: 0.8 * U, speedScale: 2.4 * U,
            note: `Two Lamb-Chaplygin dipoles, exact steady vortex pairs that translate at their speed (${U} m/s, radius 10 m, 10 m deep), meet ${offset} m off axis. Head on, the vortices exchange partners and leave sideways as two new pairs; off axis, they turn and orbit. Periodic 100 m square, low-Froude correction on.` };
    },
    // Geostrophic adjustment of a raised dome on the f-plane (Coriolis: tests/coriolis.test.mjs).
    'rotating-dam-break'({ lat, dome }) {
        const W = 200, d = 2000, n = W * W, f = 2 * 7.2921e-5 * Math.sin(lat * Math.PI / 180), H0 = 10, state = new Float32Array(4 * n), concentration = new Float32Array(n);
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
            const r = Math.hypot((x + 0.5 - W / 2) * d, (y + 0.5 - W / 2) * d), i = y * W + x;
            const inside = r < 50e3;
            state[4 * i] = H0 + (inside ? dome : 0); concentration[i] = inside ? 1 : 0;
        }
        return { config: { width: W, height: W, dx: d, dy: d, boundaryX: 'open', boundaryY: 'open', coriolis: f, tracer: true },
            initial: { state, bed: new Float32Array(n), concentration }, depthScale: 11, wetDepth: 0.01, rate: 3600,
            colour: 'speed', concentration: 'linear', vorticityScale: 2e-5, view3d: true, exaggeration: 1,
            surfaceGain: 60000 / dome, still: H0, elevationScale: dome, hideGround: true, speedScale: 0.4 * dome,
            note: `A dome of water ${dome} m high and 100 km across, on a 10 m deep ocean rotating as at ${lat}° N (f = ${f.toExponential(2)} 1/s${f > 0 ? `; Rossby radius ${(Math.sqrt(9.81 * 10) / f / 1000).toFixed(0)} km` : ''}), is released. Without rotation (latitude 0) it spreads away as a ring wave; with rotation it radiates inertia-gravity waves and settles into a geostrophic eddy turning clockwise, which keeps its water. One hour per second.` };
    },
};
