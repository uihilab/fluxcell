// Classical shallow-water flows with known answers, for tests and the viewer: a stationary hydraulic
// jump (Belanger conjugate depths), an oblique shock at a wall deflection (Ippen's relation) and the
// Lamb-Chaplygin dipole (translates at its speed U).

const G = 9.81;

// Conjugate depth of a hydraulic jump (Belanger): h2 / h1 = (sqrt(1 + 8 Fr1²) - 1) / 2.
export const conjugateDepth = (h1, fr1) => h1 * (Math.sqrt(1 + 8 * fr1 * fr1) - 1) / 2;

// Supercritical inflow from a reservoir: a stage boundary at the energy level E = h1 + u1²/2g gives
// inflow at depth h1 and velocity u1 (water entering through a stage face comes from rest at the
// stage; model.mjs: hydrographGhost).
function supercriticalInflow(h1, u1) { return h1 + u1 * u1 / (2 * G); }

// Stationary jump in a flat, frictionless channel: supercritical inflow (h1, Fr1) on the west, the
// conjugate depth held by a stage on the east; the jump starts at x = L/2 and should stay there.
export function hydraulicJumpCase(dx = 0.5, { L = 200, width = 12, h1 = 0.5, fr1 = 3 } = {}) {
    const nx = Math.round(L / dx), ny = Math.max(1, Math.round(width / dx)), n = nx * ny;
    const u1 = fr1 * Math.sqrt(G * h1), q = h1 * u1, h2 = conjugateDepth(h1, fr1), u2 = q / h2;
    const state = new Float64Array(4 * n), bed = new Float64Array(n);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const k = 4 * (y * nx + x), up = (x + 0.5) * dx < L / 2;
        state.set([up ? h1 : h2, up ? q : h2 * u2, 0, 0], k);
    }
    return {
        config: { width: nx, height: ny, dx, dy: dx, boundaryX: 'wall', boundaryY: 'wall',
            hydrographs: [{ edge: 'west', kind: 'stage', series: [[0, supercriticalInflow(h1, u1)]] },
                { edge: 'east', kind: 'stage', series: [[0, h2]] }] },
        initial: { state, bed }, h1, h2, fr1, q, L,
    };
}

// Ippen's oblique hydraulic jump: wave angle beta for a wall deflection theta at upstream Froude
// number F, from tan(theta) = tan(beta) (s - 3) / (2 tan²(beta) + s - 1), s = sqrt(1 + 8 F² sin²(beta)),
// on the weak branch (bisection between the Mach angle and the maximum deflection); depth ratio
// h2/h1 = (s - 1) / 2.
export function obliqueShock(fr, theta) {
    const deflection = beta => {
        const t = Math.tan(beta), s = Math.sqrt(1 + 8 * fr * fr * Math.sin(beta) ** 2);
        return Math.atan(t * (s - 3) / (2 * t * t + s - 1));
    };
    let lo = Math.asin(1 / fr), hi = lo;
    for (let b = lo; b < Math.PI / 2; b += 1e-4) { if (deflection(b) > deflection(hi)) hi = b; }
    if (theta > deflection(hi)) throw new RangeError('deflection beyond the attached-shock limit');
    for (let k = 0; k < 200; k++) { const m = (lo + hi) / 2; if (deflection(m) < theta) lo = m; else hi = m; }
    const beta = (lo + hi) / 2, s = Math.sqrt(1 + 8 * fr * fr * Math.sin(beta) ** 2);
    return { beta, ratio: (s - 1) / 2 };
}

// Oblique hydraulic jump at a straight wall, set up without a staircased wedge: the wall is the
// row of solid cells y = 0 from x0 on (aligned with the grid) and the supercritical inflow (h1, Fr1)
// arrives turned by theta toward it; the wall turns it back by theta, and Ippen's shock leaves the
// leading corner (x0, 0) at beta - theta to the wall. The domain is periodic in y, so upstream of x0
// the tilted inflow leaves at the bottom and re-enters at the top, staying uniform; the west edge
// (zero-gradient) sees only that inflow, crossed supercritically, and the east edge lets the flow
// out. The wake under the wall starts at the top edge (through the periodic wrap); Ly keeps its
// expansion fan (about 30 degrees below the flow) away from the shock. The initial shock is placed at a wrong angle (initialAngle, measured from the
// wall) with the post-shock state below it: the flow sweeps the error downstream at about u1 and
// leaves the steady pattern, so the angle and depth reached are a test of the model against Ippen.
export function obliqueShockCase(dx = 0.5, { Lx = 90, Ly = 70, x0 = 20, h1 = 0.5, fr1 = 3, theta = 10 * Math.PI / 180, initialAngle = 25 * Math.PI / 180 } = {}) {
    const nx = Math.round(Lx / dx), ny = Math.round(Ly / dx), n = nx * ny, u1 = fr1 * Math.sqrt(G * h1);
    const shock = obliqueShock(fr1, theta), h2 = h1 * shock.ratio;
    // Post-shock velocity, along the wall: tangential component (to the shock) conserved, normal
    // component from mass conservation across the shock.
    const ut = u1 * Math.cos(shock.beta), un = u1 * Math.sin(shock.beta) * h1 / h2, u2 = Math.hypot(ut, un);
    const state = new Float64Array(4 * n), bed = new Float64Array(n), solid = new Uint8Array(n);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const i = y * nx + x, X = (x + 0.5) * dx, Y = (y + 0.5) * dx;
        solid[i] = y === 0 && X > x0 ? 1 : 0;
        if (solid[i]) continue;
        state.set(X < x0 || Y - dx / 2 > (X - x0) * Math.tan(initialAngle) ? [h1, h1 * u1 * Math.cos(theta), -h1 * u1 * Math.sin(theta), 0] : [h2, h2 * u2, 0, 0], 4 * i);
    }
    return { config: { width: nx, height: ny, dx, dy: dx, boundaryX: 'open', boundaryY: 'periodic' },
        initial: { state, bed, solid }, h1, h2, fr1, theta, x0, ...shock };
}

// Bessel functions J0, J1 by their power series (|x| <= 10 here).
export function besselJ(nu, x) {
    let term = (x / 2) ** nu / (nu === 0 ? 1 : 1), sum = term;
    for (let k = 1; k < 60; k++) { term *= -(x * x / 4) / (k * (k + nu)); sum += term; }
    return sum;
}
const J1_ZERO = 3.8317059702075125;

// Lamb-Chaplygin dipole of radius a moving at speed U in direction angle phi, centred at (cx, cy)
// (m), lab-frame stream function: inside r < a, psi = U r sin(t) - 2U J1(k r) sin(t) / (k J0(k a));
// outside, psi = U a² sin(t) / r; t is the angle from the direction of travel, k a = 3.8317.
// Velocities u = dpsi/dy, v = -dpsi/dx (by central differences of psi, h = 1e-4 a).
export function lambDipole(x, y, { cx, cy, a, U, phi = 0 }) {
    const k = J1_ZERO / a, J0ka = besselJ(0, J1_ZERO);
    const psi = (X, Y) => {
        const dxl = X - cx, dyl = Y - cy, xr = dxl * Math.cos(phi) + dyl * Math.sin(phi), yr = -dxl * Math.sin(phi) + dyl * Math.cos(phi);
        const r = Math.hypot(xr, yr), s = r > 0 ? yr / r : 0;
        return r < a ? U * r * s - 2 * U * besselJ(1, k * r) * s / (k * J0ka) : U * a * a * s / r;
    };
    const h = 1e-4 * a;
    return [(psi(x, y + h) - psi(x, y - h)) / (2 * h), -(psi(x + h, y) - psi(x - h, y)) / (2 * h)];
}

// Dipoles in a periodic square of side L (m) and still depth h0, initialized with uniform depth
// (the low-Froude pressure adjustment radiates weak gravity waves). Each dipole carries dye (c = 1 or
// a second colour, c = 0.5) inside its radius.
export function dipoleCase(n, dipoles, { L = 100, h0 = 10, tracer = true } = {}) {
    const dx = L / n, N = n * n, state = new Float64Array(4 * N), concentration = new Float64Array(N);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const X = (x + 0.5) * dx, Y = (y + 0.5) * dx, i = y * n + x;
        let u = 0, v = 0;
        dipoles.forEach((d, j) => {
            // Nearest periodic image of the dipole centre.
            const ox = Math.round((X - d.cx) / L) * L, oy = Math.round((Y - d.cy) / L) * L;
            const [du, dv] = lambDipole(X, Y, { ...d, cx: d.cx + ox, cy: d.cy + oy });
            u += du; v += dv;
            if (Math.hypot(X - d.cx - ox, Y - d.cy - oy) < d.a) concentration[i] = j % 2 ? 0.5 : 1;
        });
        state.set([h0, h0 * u, h0 * v, 0], 4 * i);
    }
    return { config: { width: n, height: n, dx, dy: dx, boundaryX: 'periodic', boundaryY: 'periodic', tracer },
        initial: { state, bed: new Float64Array(N), ...(tracer ? { concentration } : {}) } };
}

// Centroid (m) of positive (or negative) vorticity, from the state of a periodic square grid,
// unwrapped around a reference point.
export function vorticityCentroid(state, n, dx, sign, ref) {
    const L = n * dx; let sw = 0, sx = 0, sy = 0;
    const vel = (x, y) => { const i = ((y + n) % n) * n + ((x + n) % n), h = state[4 * i]; return [state[4 * i + 1] / h, state[4 * i + 2] / h]; };
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const w = sign * ((vel(x + 1, y)[1] - vel(x - 1, y)[1]) - (vel(x, y + 1)[0] - vel(x, y - 1)[0])) / (2 * dx);
        if (!(w > 0)) continue;
        const X = (x + 0.5) * dx, Y = (y + 0.5) * dx;
        sw += w; sx += w * (X - Math.round((X - ref[0]) / L) * L); sy += w * (Y - Math.round((Y - ref[1]) / L) * L);
    }
    return [sx / sw, sy / sw];
}
