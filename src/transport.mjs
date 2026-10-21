// Scalar transport and shallow-flow dynamics cases with exact solutions (SI units). Each case
// returns { config, initial } for the solvers and, where one exists, exact(t) giving cell averages
// { depth, concentration } on the same grid. Shared by the CPU tests and the GPU validation page.

// erf to ~1e-15: Taylor series for |x| < 3, continued fraction for erfc beyond.
export function erf(x) {
    const a = Math.abs(x);
    if (a < 3) {
        let term = a, sum = a;
        for (let n = 1; n < 200 && Math.abs(term) > 1e-17 * Math.abs(sum); n++) {
            term *= -a * a / n;
            sum += term / (2 * n + 1);
        }
        return Math.sign(x) * 2 / Math.sqrt(Math.PI) * sum;
    }
    let f = 0;
    for (let n = 60; n >= 1; n--) f = n / 2 / (a + f);
    const erfc = Math.exp(-a * a) / Math.sqrt(Math.PI) / (a + f);
    return Math.sign(x) * (1 - erfc);
}

// Five-point Gauss-Legendre nodes and weights on [-1/2, 1/2] (weights sum to 1).
const GAUSS = [[-0.4530899229693320, 0.1184634425280945], [-0.2692346550528415, 0.2393143352496832], [0, 0.2844444444444444],
    [0.2692346550528415, 0.2393143352496832], [0.4530899229693320, 0.1184634425280945]];
function cellAverage1d(f, x0, dx) { return GAUSS.reduce((s, [t, w]) => s + w * f(x0 + (0.5 + t) * dx), 0); }
function cellAverage2d(f, x0, y0, dx, dy) {
    let s = 0;
    for (const [tx, wx] of GAUSS) for (const [ty, wy] of GAUSS) s += wx * wy * f(x0 + (0.5 + tx) * dx, y0 + (0.5 + ty) * dy);
    return s;
}
const wrap = (x, L) => ((x % L) + L) % L;

// Uniform flow (h = 1 m, u = 1 m/s) along a periodic 10 m channel carrying a Gaussian (sigma 0.75 m)
// or a 3 m square pulse; the exact concentration is the initial profile translated by u t.
export function advectionCase(n, shape = 'gaussian', { tracer = true } = {}) {
    const L = 10, h = 1, u = 1, dx = L / n, rows = 2;
    const profile = shape === 'gaussian' ? x => Math.exp(-((x - 5) ** 2) / (2 * 0.75 ** 2))
        : x => (x >= 3.5 && x < 6.5 ? 1 : 0);
    const state = new Float64Array(4 * n * rows), concentration = new Float64Array(n * rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x;
        state.set([h, h * u, 0, 0], 4 * i);
        concentration[i] = shape === 'gaussian' ? cellAverage1d(profile, x * dx, dx)
            : Math.max(0, Math.min(6.5, (x + 1) * dx) - Math.max(3.5, x * dx)) / dx;
    }
    const config = { width: n, height: rows, dx, dy: dx, boundaryX: 'periodic', boundaryY: 'periodic', tracer };
    const exact = t => {
        const c = new Float64Array(n * rows);
        if (shape === 'gaussian') for (let y = 0; y < rows; y++) for (let x = 0; x < n; x++) {
            c[y * n + x] = cellAverage1d(s => profile(wrap(s - u * t, L)), x * dx, dx);
        } else {
            // Square pulse: exact cell averages of the translated indicator.
            for (let x = 0; x < n; x++) {
                let covered = 0;
                for (const shift of [-L, 0, L]) {
                    const a = 3.5 + u * t + shift, b = 6.5 + u * t + shift;
                    covered += Math.max(0, Math.min(b, (x + 1) * dx) - Math.max(a, x * dx));
                }
                for (let y = 0; y < rows; y++) c[y * n + x] = covered / dx;
            }
        }
        return { concentration: c };
    };
    return { config, initial: { state, bed: new Float64Array(n * rows), concentration }, exact };
}

// Still water (h = 1 m) in a periodic 10 m channel; a Gaussian (sigma0 0.5 m) spreads with
// diffusivity D: c = sigma0/sigma(t) exp(-(x - 5)^2 / (2 sigma(t)^2)), sigma^2 = sigma0^2 + 2 D t,
// summed over periodic images.
export function diffusionCase(n, D = 0.01) {
    const L = 10, dx = L / n, rows = 2, s0 = 0.5;
    const field = t => {
        const s2 = s0 * s0 + 2 * D * t;
        return x => { let v = 0; for (let k = -3; k <= 3; k++) v += Math.exp(-((x - 5 + k * L) ** 2) / (2 * s2)); return v * s0 / Math.sqrt(s2); };
    };
    const state = new Float64Array(4 * n * rows), concentration = new Float64Array(n * rows), f0 = field(0);
    for (let y = 0; y < rows; y++) for (let x = 0; x < n; x++) {
        state[4 * (y * n + x)] = 1;
        concentration[y * n + x] = cellAverage1d(f0, x * dx, dx);
    }
    const config = { width: n, height: rows, dx, dy: dx, boundaryX: 'periodic', boundaryY: 'periodic', tracer: true, diffusivity: D };
    const exact = t => {
        const f = field(t), c = new Float64Array(n * rows);
        for (let y = 0; y < rows; y++) for (let x = 0; x < n; x++) c[y * n + x] = cellAverage1d(f, x * dx, dx);
        return { concentration: c };
    };
    return { config, initial: { state, bed: new Float64Array(n * rows), concentration }, exact };
}

// Stationary Gaussian vortex in cyclostrophic balance on a flat bed (100 m square, walls):
// u_theta = V (r/R) exp((1 - r^2/R^2)/2), h = h_inf - V^2 e / (2 g) exp(-r^2/R^2), an exact steady
// solution. A tracer front c0 = (1 + tanh(y'/w))/2 through the centre is wound up by the angular
// velocity Omega(r) = u_theta / r; the exact concentration is c0 rotated back by Omega(r) t.
export function vortexCase(n, { V = 1, R = 10, hInf = 1, w = 3, gravity = 9.81, tracer = true } = {}) {
    const L = 100, dx = L / n, c = L / 2;
    const speed = r => V * (r / R) * Math.exp((1 - (r * r) / (R * R)) / 2);
    const depth = (x, y) => hInf - V * V * Math.E / (2 * gravity) * Math.exp(-((x - c) ** 2 + (y - c) ** 2) / (R * R));
    const front = (x, y, t) => {
        const dxp = x - c, dyp = y - c, r = Math.hypot(dxp, dyp);
        const angle = r > 0 ? -(speed(r) / r) * t : 0;
        const yr = Math.sin(angle) * dxp + Math.cos(angle) * dyp;
        return 0.5 * (1 + Math.tanh(yr / w));
    };
    const N = n * n, state = new Float64Array(4 * N), concentration = new Float64Array(N);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x, x0 = x * dx, y0 = y * dx;
        // Cell averages of h, h u and h v; the concentration is the h-weighted average.
        const h = cellAverage2d(depth, x0, y0, dx, dx);
        const qx = cellAverage2d((X, Y) => { const r = Math.hypot(X - c, Y - c); return r > 0 ? -depth(X, Y) * speed(r) * (Y - c) / r : 0; }, x0, y0, dx, dx);
        const qy = cellAverage2d((X, Y) => { const r = Math.hypot(X - c, Y - c); return r > 0 ? depth(X, Y) * speed(r) * (X - c) / r : 0; }, x0, y0, dx, dx);
        state.set([h, qx, qy, 0], 4 * i);
        concentration[i] = cellAverage2d((X, Y) => depth(X, Y) * front(X, Y, 0), x0, y0, dx, dx) / h;
    }
    const config = { width: n, height: n, dx, dy: dx, boundaryX: 'wall', boundaryY: 'wall', gravity, tracer };
    const exact = t => {
        const depthAvg = new Float64Array(N), conc = new Float64Array(N);
        for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
            const i = y * n + x;
            depthAvg[i] = state[4 * i];
            conc[i] = cellAverage2d((X, Y) => depth(X, Y) * front(X, Y, t), x * dx, y * dx, dx, dx) / depthAvg[i];
        }
        return { depth: depthAvg, concentration: conc, speed };
    };
    return { config, initial: { state, bed: new Float64Array(N), concentration }, exact, centre: c, R, V };
}

// A cyclostrophically balanced Gaussian vortex (swirl V r/R exp((1 - r²/R²)/2), R = 3 m) carried by
// a 0.5 m/s stream (2 m deep) out through a stage boundary in a 60 m x 40 m channel (discharge in on
// the west, walls north and south). sign = -1 is clockwise; yc is the distance of the centre from the
// south wall. The vortex reaches the outlet after about 40 s; after it has left, the swirl energy
// (kinetic energy relative to the stream) should be a small fraction of the initial one.
export function vortexExitCase(dx, { V = 1.2, sign = -1, yc = 8, R = 3, U = 0.5, h0 = 2, gravity = 9.81 } = {}) {
    const Lx = 60, Ly = 40, xc = 40, W = Math.round(Lx / dx), H = Math.round(Ly / dx), N = W * H;
    const state = new Float64Array(4 * N);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const X = (x + 0.5) * dx - xc, Y = (y + 0.5) * dx - yc, r2 = X * X + Y * Y;
        const omega = sign * V / R * Math.exp((1 - r2 / (R * R)) / 2);
        const h = h0 - V * V * Math.E / (2 * gravity) * Math.exp(-r2 / (R * R));
        state.set([h, h * (U - omega * Y), h * omega * X, 0], 4 * (y * W + x));
    }
    const config = { width: W, height: H, dx, dy: dx, boundaryX: 'wall', boundaryY: 'wall', gravity,
        hydrographs: [{ edge: 'west', kind: 'discharge', series: [[0, U * h0 * Ly]] }, { edge: 'east', kind: 'stage', series: [[0, h0]] }] };
    // Kinetic energy relative to the stream, J/(kg/m³) x m (integral of h |u - U|² / 2).
    const swirlEnergy = s => {
        let e = 0;
        for (let i = 0; i < N; i++) { const h = s[4 * i]; if (h > 0) e += ((s[4 * i + 1] - h * U) ** 2 + s[4 * i + 2] ** 2) / (2 * h); }
        return e * dx * dx;
    };
    return { config, initial: { state, bed: new Float64Array(N) }, swirlEnergy, stream: U };
}

// Steady mixing layer: uniform flow (h0 = 1 m, u = 0.5 m/s) enters a 200 m x 40 m channel through
// two discharge segments on the west edge carrying c = 1 (north half) and c = 0 (south half) and
// leaves at a fixed stage on the east edge. With diffusivity D the steady concentration is
// c = erfc(-(y - y0) / sqrt(4 D x / u)) / 2 (x from the inflow face, y0 the splitter).
export function mixingLayerCase(nx, { D = 0.02, h0 = 1, u = 0.5 } = {}) {
    const Lx = 200, Ly = 40, dx = Lx / nx, ny = Math.round(nx * Ly / Lx), dy = Ly / ny, half = ny / 2;
    const q = u * h0 * (Ly / 2); // m³/s per segment
    const config = { width: nx, height: ny, dx, dy, boundaryX: 'wall', boundaryY: 'wall', tracer: true, diffusivity: D,
        hydrographs: [
            { edge: 'west', start: 0, end: half, kind: 'discharge', series: [[0, q]], concentration: 0 },
            { edge: 'west', start: half, end: ny, kind: 'discharge', series: [[0, q]], concentration: 1 },
            { edge: 'east', kind: 'stage', series: [[0, h0]] }] };
    const N = nx * ny, state = new Float64Array(4 * N);
    for (let i = 0; i < N; i++) state.set([h0, h0 * u, 0, 0], 4 * i);
    const exact = () => {
        const c = new Float64Array(N);
        for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
            const width = Math.sqrt(4 * D * ((x + 0.5) * dx) / u);
            c[y * nx + x] = cellAverage1d(Y => 0.5 * (1 + erf((Y - Ly / 2) / width)), y * dy, dy);
        }
        return { concentration: c };
    };
    return { config, initial: { state, bed: new Float64Array(N), concentration: new Float64Array(N) }, exact };
}

// Kelvin-Helmholtz shear layer: u = U tanh((y - L/2)/delta) in a 100 m square (periodic in x,
// walls in y) of uniform depth, a small transverse perturbation of `modes` wavelengths across the
// domain, and a tracer c = (1 + tanh((y - L/2)/delta))/2 marking the two streams. The default
// delta = 0.4446 L / (2 pi modes) puts the perturbation at the most unstable wavenumber of the
// incompressible tanh layer, alpha delta = 0.4446, whose growth rate is 0.1897 U/delta (Michalke,
// 1964); at the default Froude number U / sqrt(g h) = 0.1 the shallow-water value is close to it.
export const MICHALKE = Object.freeze({ wavenumber: 0.4446, growthRate: 0.1897 });
// width: channel width across the layer (m; default the full square).
export function shearLayerCase(n, { U = 1, depth = 10, amplitude = 1e-3, modes = 4, delta = MICHALKE.wavenumber * 100 / (2 * Math.PI * modes), width = 100 } = {}) {
    const L = 100, dx = L / n, ny = Math.round(width / dx), N = n * ny, state = new Float64Array(4 * N), concentration = new Float64Array(N);
    for (let y = 0; y < ny; y++) for (let x = 0; x < n; x++) {
        const X = (x + 0.5) * dx, Y = (y + 0.5) * dx, s = (Y - ny * dx / 2) / delta, i = y * n + x;
        const v = amplitude * U * Math.sin(2 * Math.PI * modes * X / L) / Math.cosh(s);
        state.set([depth, depth * U * Math.tanh(s), depth * v, 0], 4 * i);
        concentration[i] = 0.5 * (1 + Math.tanh(s));
    }
    const config = { width: n, height: ny, dx, dy: dx, boundaryX: 'periodic', boundaryY: 'wall', tracer: true };
    return { config, initial: { state, bed: new Float64Array(N), concentration }, U, delta, modes, L };
}

// Amplitude of Fourier mode m (in x) of the transverse velocity, RMS over rows (m/s).
export function modeAmplitude(state, n, m) {
    let total = 0;
    for (let y = 0; y < n; y++) {
        let re = 0, im = 0;
        for (let x = 0; x < n; x++) {
            const k = 4 * (y * n + x), v = state[k] > 0 ? state[k + 2] / state[k] : 0, a = 2 * Math.PI * m * (x + 0.5) / n;
            re += v * Math.cos(a); im += v * Math.sin(a);
        }
        total += (re * re + im * im) * 4 / (n * n);
    }
    return Math.sqrt(total / n);
}

// In-place radix-2 complex FFT (length a power of two).
function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let cr = 1, ci = 0;
            for (let j = 0; j < len / 2; j++) {
                const a = i + j, b = a + len / 2;
                const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
                [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
            }
        }
    }
}

// Shell-averaged kinetic energy spectrum E(k) of a doubly periodic n x n velocity field
// (k in units of 2 pi / L, k = 0 .. n/2), normalized so that sum_k E(k) = mean (u² + v²) / 2.
export function energySpectrum(state, n) {
    const E = new Float64Array(n / 2 + 1);
    for (const c of [1, 2]) {
        const re = new Float64Array(n * n), im = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) re[i] = state[4 * i] > 0 ? state[4 * i + c] / state[4 * i] : 0;
        const row = [new Float64Array(n), new Float64Array(n)];
        for (let y = 0; y < n; y++) {
            row[0].set(re.subarray(y * n, y * n + n)); row[1].fill(0);
            fft(row[0], row[1]);
            re.set(row[0], y * n); im.set(row[1], y * n);
        }
        for (let x = 0; x < n; x++) {
            for (let y = 0; y < n; y++) { row[0][y] = re[y * n + x]; row[1][y] = im[y * n + x]; }
            fft(row[0], row[1]);
            for (let y = 0; y < n; y++) { re[y * n + x] = row[0][y]; im[y * n + x] = row[1][y]; }
        }
        for (let ky = 0; ky < n; ky++) for (let kx = 0; kx < n; kx++) {
            const k = Math.round(Math.hypot(kx <= n / 2 ? kx : kx - n, ky <= n / 2 ? ky : ky - n));
            if (k <= n / 2) { const i = ky * n + kx; E[k] += 0.5 * (re[i] * re[i] + im[i] * im[i]) / (n * n * n * n); }
        }
    }
    return E;
}

// Domain means of kinetic energy (u² + v²)/2 and enstrophy w²/2 (periodic central differences).
export function energyEnstrophy(state, n, dx) {
    const w = vorticity(state, n, n, dx, dx, true);
    let e = 0, z = 0;
    for (let i = 0; i < n * n; i++) {
        const h = state[4 * i], u = h > 0 ? state[4 * i + 1] / h : 0, v = h > 0 ? state[4 * i + 2] / h : 0;
        e += 0.5 * (u * u + v * v); z += 0.5 * w[i] * w[i];
    }
    return { energy: e / (n * n), enstrophy: z / (n * n) };
}

// Mixing measure: depth-weighted mean of 4 c (1 - c) (0 for unmixed water, 1 for an even mix).
export function mixedFraction(state, amount) {
    let mixed = 0, volume = 0;
    for (let i = 0; i < amount.length; i++) {
        const h = state[4 * i];
        if (!(h > 0)) continue;
        const c = Math.min(1, Math.max(0, amount[i] / h));
        mixed += 4 * h * c * (1 - c); volume += h;
    }
    return volume > 0 ? mixed / volume : 0;
}

// L1 norm of the difference between two cell fields, as a mean over cells.
export function meanAbsoluteError(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length;
}

// Shallow wake: uniform inflow (h0 = 2 m, u = 0.5 m/s) past a circular island of diameter
// 10 m in a 400 m x 100 m channel (discharge in on the west, fixed stage on the east, free-slip
// side walls). A dye band of c = 1 enters around the centreline. The island sits 0.3 cells off the
// centreline so that shedding starts from a deterministic asymmetry.
export function wakeCase(dx = 0.5, { u = 0.5, h0 = 2, D = 10, manning = 0 } = {}) {
    const Lx = 400, Ly = 100, nx = Math.round(Lx / dx), ny = Math.round(Ly / dx), N = nx * ny;
    const cx = 80, cy = Ly / 2 + 0.3 * dx;
    const solid = new Uint8Array(N), state = new Float64Array(4 * N);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const i = y * nx + x, X = (x + 0.5) * dx, Y = (y + 0.5) * dx;
        solid[i] = (X - cx) ** 2 + (Y - cy) ** 2 < (D / 2) ** 2 ? 1 : 0;
        if (!solid[i]) state.set([h0, h0 * u, 0, 0], 4 * i);
    }
    const band = [Math.round((Ly / 2 - 6) / dx), Math.round((Ly / 2 + 6) / dx)];
    const q = width => u * h0 * width * dx;
    const config = { width: nx, height: ny, dx, dy: dx, boundaryX: 'wall', boundaryY: 'wall', tracer: true,
        hydrographs: [
            { edge: 'west', start: 0, end: band[0], kind: 'discharge', series: [[0, q(band[0])]], concentration: 0 },
            { edge: 'west', start: band[0], end: band[1], kind: 'discharge', series: [[0, q(band[1] - band[0])]], concentration: 1 },
            { edge: 'west', start: band[1], end: ny, kind: 'discharge', series: [[0, q(ny - band[1])]], concentration: 0 },
            { edge: 'east', kind: 'stage', series: [[0, h0]] }] };
    const initial = { state, bed: new Float64Array(N), solid, concentration: new Float64Array(N),
        ...(manning > 0 ? { manning: new Float64Array(N).fill(manning) } : {}) };
    // Probe 4 diameters downstream on the centreline (transverse velocity gives the shedding frequency).
    const probe = Math.round(Ly / 2 / dx) * nx + Math.round((cx + 4 * D) / dx);
    return { config, initial, probe, diameter: D, velocity: u };
}

// Seeded uniform random numbers (mulberry32).
export function random(seed) {
    return () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Decaying two-dimensional turbulence in a doubly periodic 100 m square of 10 m depth (Froude
// number about 0.05): a non-divergent random velocity field from a stream function whose energy
// sits in wavenumbers 6 to 10 (in units of 2 pi / L), with RMS speed urms. The tracer starts as
// c = 1 in the west half and 0 in the east half.
export function turbulenceCase(n, { urms = 0.5, depth = 10, seed = 7, kmin = 6, kmax = 10 } = {}) {
    const L = 100, dx = L / n, N = n * n, rand = random(seed), modes = [];
    for (let kx = -kmax; kx <= kmax; kx++) for (let ky = 0; ky <= kmax; ky++) {
        const k = Math.hypot(kx, ky);
        if (k < kmin || k > kmax || (ky === 0 && kx <= 0)) continue;
        modes.push([kx, ky, 2 * Math.PI * rand(), 1 / k ** 2]); // amplitude: flat energy shell
    }
    const u = new Float64Array(N), v = new Float64Array(N);
    const cosx = new Float64Array(n), sinx = new Float64Array(n);
    for (const [kx, ky, phase, a] of modes) {
        // psi = a cos(kx X + ky Y + phase); u = dpsi/dy, v = -dpsi/dx (cell centres).
        const w = 2 * Math.PI / L;
        for (let x = 0; x < n; x++) { cosx[x] = Math.cos(w * kx * (x + 0.5) * dx); sinx[x] = Math.sin(w * kx * (x + 0.5) * dx); }
        for (let y = 0; y < n; y++) {
            const ay = w * ky * (y + 0.5) * dx + phase, cy = Math.cos(ay), sy = Math.sin(ay);
            for (let x = 0; x < n; x++) {
                const s = sinx[x] * cy + cosx[x] * sy; // sin(kx X + ky Y + phase)
                u[y * n + x] += -a * w * ky * s; v[y * n + x] += a * w * kx * s;
            }
        }
    }
    let energy = 0;
    for (let i = 0; i < N; i++) energy += u[i] * u[i] + v[i] * v[i];
    const scale = urms / Math.sqrt(energy / N);
    const state = new Float64Array(4 * N), concentration = new Float64Array(N);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x;
        state.set([depth, depth * u[i] * scale, depth * v[i] * scale, 0], 4 * i);
        concentration[i] = x < n / 2 ? 1 : 0;
    }
    const config = { width: n, height: n, dx, dy: dx, boundaryX: 'periodic', boundaryY: 'periodic', tracer: true };
    return { config, initial: { state, bed: new Float64Array(N), concentration } };
}

// Relative vorticity (1/s) at cell centres by central differences (periodic or clamped edges).
export function vorticity(state, width, height, dx, dy, periodic = true) {
    const w = new Float64Array(width * height);
    const at = (x, y) => {
        x = periodic ? (x + width) % width : Math.max(0, Math.min(width - 1, x));
        y = periodic ? (y + height) % height : Math.max(0, Math.min(height - 1, y));
        return 4 * (y * width + x);
    };
    const vel = (k, c) => (state[k] > 1e-6 ? state[k + c] / state[k] : 0);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        w[y * width + x] = (vel(at(x + 1, y), 2) - vel(at(x - 1, y), 2)) / (2 * dx) - (vel(at(x, y + 1), 1) - vel(at(x, y - 1), 1)) / (2 * dy);
    }
    return w;
}

// Viscous shear layer with a constant eddy viscosity nu: in uniform depth h = 1 m with v = 0, the
// x momentum obeys u_t = nu u_yy, so u = U erf((y - L/2) / sqrt(4 nu (t + t0))) exactly. Periodic in
// x (4 cells), free-slip walls in y at +-10 m (the profile is flat there to 1e-8).
export function viscousShearCase(n, { nu = 0.05, U = 0.2, t0 = 10 } = {}) {
    const L = 20, dy = L / n, w = 4, N = w * n, state = new Float64Array(4 * N);
    const profile = t => Y => U * erf((Y - L / 2) / Math.sqrt(4 * nu * (t + t0)));
    const f0 = profile(0);
    for (let y = 0; y < n; y++) for (let x = 0; x < w; x++) state.set([1, cellAverage1d(f0, y * dy, dy), 0, 0], 4 * (y * w + x));
    const config = { width: w, height: n, dx: dy, dy, boundaryX: 'periodic', boundaryY: 'wall', eddyViscosity: nu };
    const exact = t => {
        const f = profile(t), u = new Float64Array(N);
        for (let y = 0; y < n; y++) for (let x = 0; x < w; x++) u[y * w + x] = cellAverage1d(f, y * dy, dy);
        return { velocity: u };
    };
    return { config, initial: { state, bed: new Float64Array(N) }, exact };
}
