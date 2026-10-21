// SI units. State is interleaved [h, hu, hv, reserved] in row-major order.
export const STATE_STRIDE = 4;
// GPU boundary codes: wall=0, periodic=1, open (zero-gradient, transmissive)=2,
// outflow (free outfall: zero-gradient when flow leaves, reflective when it would enter)=3.
export const BOUNDARIES = Object.freeze(['wall', 'periodic', 'open', 'outflow']);
export const boundaryCode = name => BOUNDARIES.indexOf(name);

export function configuration(options) {
    const p = { dx: 1, dy: 1, gravity: 9.81, cfl: 0.4, rain: 0, datum: 'auto',
        boundaryX: 'wall', boundaryY: 'wall', tracer: false, diffusivity: 0, lowFroude: false,
        eddyViscosity: 0, smagorinsky: 0, bedTurbulence: 0, schmidt: 0.7, wind: [0, 0], windDrag: 'wu1982', settling: null, coriolis: 0, ...options };
    for (const key of ['width', 'height']) {
        if (!Number.isSafeInteger(p[key]) || p[key] < 1 || p[key] > 65535) {
            throw new RangeError(`${key} must be an integer in [1, 65535]`);
        }
    }
    for (const key of ['dx', 'dy', 'gravity', 'cfl']) {
        if (!Number.isFinite(p[key]) || p[key] <= 0) throw new RangeError(`${key} must be positive and finite`);
    }
    if (p.cfl > 0.4) throw new RangeError('cfl must not exceed 0.4');
    if (!Number.isFinite(p.rain) || p.rain < 0) throw new RangeError('rain must be finite and nonnegative (m/s)');
    for (const key of ['boundaryX', 'boundaryY']) {
        if (!BOUNDARIES.includes(p[key])) throw new RangeError(`${key} must be wall, periodic, open or outflow`);
    }
    if (p.datum !== 'auto' && !Number.isFinite(p.datum)) throw new RangeError("datum must be 'auto' or a finite elevation (m)");
    if (p.width * p.height > 0x3fffffff) throw new RangeError('Grid exceeds the state indexing range');
    if (typeof p.tracer !== 'boolean') throw new RangeError('tracer must be true or false');
    if (typeof p.lowFroude !== 'boolean') throw new RangeError('lowFroude must be true or false');
    if (!Number.isFinite(p.diffusivity) || p.diffusivity < 0) throw new RangeError('diffusivity must be finite and nonnegative (m²/s)');
    if (p.diffusivity > 0 && !p.tracer) throw new RangeError('diffusivity requires tracer: true');
    if (!Number.isFinite(p.eddyViscosity) || p.eddyViscosity < 0) throw new RangeError('eddyViscosity must be finite and nonnegative (m²/s)');
    if (!Number.isFinite(p.smagorinsky) || p.smagorinsky < 0 || p.smagorinsky > MAX_SMAGORINSKY) {
        throw new RangeError(`smagorinsky must lie in [0, ${MAX_SMAGORINSKY}]`);
    }
    if (!Number.isFinite(p.bedTurbulence) || p.bedTurbulence < 0 || p.bedTurbulence > 1) throw new RangeError('bedTurbulence must lie in [0, 1]');
    if (!Number.isFinite(p.schmidt) || p.schmidt <= 0) throw new RangeError('schmidt must be positive');
    if (!Array.isArray(p.wind) || p.wind.length !== 2 || !p.wind.every(Number.isFinite) || Math.hypot(...p.wind) > MAX_WIND) {
        throw new RangeError(`wind must be [x, y] (m/s at 10 m) with speed at most ${MAX_WIND} m/s`);
    }
    if (p.windDrag !== 'wu1982' && !(Number.isFinite(p.windDrag) && p.windDrag > 0 && p.windDrag <= 0.01)) {
        throw new RangeError("windDrag must be 'wu1982' or a drag coefficient in (0, 0.01]");
    }
    p.wind = Object.freeze([...p.wind]);
    if (!Number.isFinite(p.coriolis) || Math.abs(p.coriolis) > 1e-3) throw new RangeError('coriolis (f, 1/s) must be finite with |f| <= 1e-3');
    if (p.settling !== null) {
        const { velocity, exponent = 0 } = p.settling ?? {};
        if (!p.tracer) throw new RangeError('settling requires tracer: true');
        if (!(Number.isFinite(velocity) && velocity >= 0 && velocity <= 1)) throw new RangeError('settling.velocity must lie in [0, 1]');
        if (!(Number.isFinite(exponent) && exponent >= 0 && exponent <= 3)) throw new RangeError('settling.exponent must lie in [0, 3]');
        p.settling = Object.freeze({ velocity, exponent });
    }
    p.hydrographs = validateHydrographs(p, p.hydrographs ?? []);
    return Object.freeze(p);
}

// Wind stress: uniform wind W = [x, y] (m/s at 10 m) acting on the water surface. The kinematic
// stress tau / rho_w = (rho_a / rho_w) C_d |W| W (m²/s²) is added to the momentum of wet cells
// (h > H_THIN) after the flux update and before friction: q += dt tau / rho_w. C_d is a constant
// (windDrag) or, by default, the linear law of Wu (1982), C_d = (0.8 + 0.065 |W|) 1e-3 for |W| in
// m/s. Air 1.225 kg/m³, water 1000 kg/m³.
export const MAX_WIND = 100;

// Coriolis: dq/dt = -f k x q (f > 0 in the northern hemisphere; f = 2 Omega sin(latitude)). Applied to
// wet cells after the wind stress as the exact rotation of q by the angle -f dt, which conserves |q|
// (kinetic energy) and reproduces the inertial oscillation exactly at fixed depth.
export function coriolisRotate(f, dt, qx, qy) {
    if (f === 0) return [qx, qy];
    const c = Math.cos(f * dt), s = Math.sin(f * dt);
    return [c * qx + s * qy, -s * qx + c * qy];
}
export const AIR_WATER_DENSITY_RATIO = 1.225 / 1000;
export function windDragCoefficient(p, speed) {
    return p.windDrag === 'wu1982' ? (0.8 + 0.065 * speed) * 1e-3 : p.windDrag;
}
export function windStress(p) {
    const speed = Math.hypot(p.wind[0], p.wind[1]), k = AIR_WATER_DENSITY_RATIO * windDragCoefficient(p, speed) * speed;
    return [k * p.wind[0], k * p.wind[1]];
}

// Settling of the tracer (suspended sediment, concentration c in kg/m³): d(h c)/dt = -w_s c, with
// w_s = velocity * c^exponent (m/s). exponent = 0 is a constant settling velocity; exponent > 0 is the
// power law of the flocculation range of cohesive sediment (Mehta, 1986), with site-specific
// coefficients. Each step applies the exact solution of dc/dt = -w_s(c) c / h at the new depth h:
// c exp(-velocity dt / h), or c (1 + exponent velocity dt c^exponent / h)^(-1/exponent); it is
// positive, monotone and never overshoots. Settled mass leaves the water column (deposition).
export function settledConcentration(settling, c, h, dt) {
    if (!settling || !(c > 0) || !(h > 0) || !(settling.velocity > 0)) return c;
    const { velocity, exponent } = settling;
    if (exponent === 0) return c * Math.exp(-velocity * dt / h);
    return c * (1 + exponent * velocity * dt * c ** exponent / h) ** (-1 / exponent);
}

// Horizontal turbulence closure (depth-averaged eddy viscosity), off unless a coefficient is set:
// nu = eddyViscosity (constant background, m²/s) + (smagorinsky * min(dx, dy))² |S| (Smagorinsky)
// + bedTurbulence * u* h (bed-generated, u* = sqrt(g) n |u| / h^(1/6) from Manning; kappa/6 = 0.067
// is the classical value). |S| = sqrt(2 u_x² + 2 v_y² + (u_y + v_x)²) by central differences. The
// momentum receives viscous fluxes -h_f nu_f grad u (Laplacian form) across interior faces, and the
// tracer a turbulent diffusivity nu_f / schmidt on top of `diffusivity`.
// Stability: with the filter width min(dx, dy), central-difference strain rates obey
// |S| <= 2 sqrt(2) umax / min(dx, dy) and the CFL bound gives dt <= 0.4 min(dx, dy) / umax, so the
// Smagorinsky diffusion number 2 nu dt (1/dx² + 1/dy²) is at most 4.53 C_s², 0.41 for C_s = 0.3, on
// any grid aspect ratio; with the advective fraction of at most 0.4 the update stays a convex
// combination. The cell-local parts (background and bed) enter the timestep through
// viscousSpeed, added to the wave speeds.
export const MAX_SMAGORINSKY = 0.3;
export const viscous = p => p.eddyViscosity > 0 || p.smagorinsky > 0 || p.bedTurbulence > 0;

// Cell-local eddy viscosity (background + bed-generated) of depth h, speed |u| and Manning n.
export function localEddyViscosity(p, h, speed, n) {
    if (!(h > H_THIN)) return 0;
    return p.eddyViscosity + (p.bedTurbulence > 0 && n > 0 ? p.bedTurbulence * Math.sqrt(p.gravity) * n * speed * h ** (5 / 6) : 0);
}

// Speed added to both wave-speed maxima so that the CFL step also satisfies
// nu_eff dt (1/dx² + 1/dy²) <= 1/4 for the cell-local viscosity (and its tracer diffusivity).
export function viscousSpeed(p, nuLocal) {
    const nu = nuLocal * (p.tracer ? Math.max(1, 1 / p.schmidt) : 1);
    return 4 * p.cfl * nu * (1 / (p.dx * p.dx) + 1 / (p.dy * p.dy)) / (1 / p.dx + 1 / p.dy);
}

// Hydrograph boundaries: edge segments with a prescribed stage (water-surface elevation, m, same
// vertical datum as the bed), a prescribed inflow discharge (m³/s over the segment, >= 0), or an
// incident wave ('wave': the incident water-surface elevation, m, on still water at `still`, m; a
// characteristic boundary that generates the incident wave and lets outgoing waves leave).
// series = [[t, value], ...] with strictly increasing t (s); values are interpolated linearly
// and held constant outside the series. A segment overrides its edge's boundary type on
// [start, end) cells along the edge (west/east: rows, south/north: columns). With a tracer, water
// entering through a segment carries its constant `concentration` (default 0).
export const EDGES = Object.freeze(['west', 'east', 'south', 'north']);
export const HYDROGRAPH_KINDS = Object.freeze(['stage', 'discharge', 'wave']);
const MAX_HYDROGRAPH_POINTS = 1 << 20;

function validateHydrographs(p, list) {
    if (!Array.isArray(list)) throw new RangeError('hydrographs must be an array');
    let points = 0;
    const used = new Map();
    const segments = list.map((segment, index) => {
        const { edge, kind } = segment ?? {};
        if (!EDGES.includes(edge)) throw new RangeError(`hydrographs[${index}].edge must be west, east, south or north`);
        if (!HYDROGRAPH_KINDS.includes(kind)) throw new RangeError(`hydrographs[${index}].kind must be stage, discharge or wave`);
        const vertical = edge === 'west' || edge === 'east';
        if ((vertical ? p.boundaryX : p.boundaryY) === 'periodic') throw new RangeError(`hydrographs[${index}] lies on a periodic edge`);
        const length = vertical ? p.height : p.width;
        const start = segment.start ?? 0, end = segment.end ?? length;
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end > length || start >= end) {
            throw new RangeError(`hydrographs[${index}] needs integer cells 0 <= start < end <= ${length}`);
        }
        const mask = used.get(edge) ?? new Uint8Array(length);
        for (let k = start; k < end; k++) {
            if (mask[k]) throw new RangeError(`hydrographs[${index}] overlaps another segment on the ${edge} edge`);
            mask[k] = 1;
        }
        used.set(edge, mask);
        // Accept an already validated segment (times/values) as well as a [t, value] series.
        const series = segment.series ?? (segment.times ? Array.from(segment.times, (t, k) => [t, segment.values[k]]) : undefined);
        if (!Array.isArray(series) || series.length < 1) throw new RangeError(`hydrographs[${index}].series needs at least one [t, value] pair`);
        const times = new Float64Array(series.length), values = new Float64Array(series.length);
        series.forEach((pair, k) => {
            const [t, v] = pair ?? [];
            if (!Number.isFinite(t) || !Number.isFinite(v)) throw new RangeError(`hydrographs[${index}].series[${k}] is not a finite [t, value] pair`);
            if (k > 0 && !(t > times[k - 1])) throw new RangeError(`hydrographs[${index}].series times must increase strictly`);
            if (kind === 'discharge' && v < 0) throw new RangeError(`hydrographs[${index}] discharge must be nonnegative`);
            times[k] = t; values[k] = v;
        });
        const concentration = segment.concentration ?? 0;
        if (!Number.isFinite(concentration) || concentration < 0) throw new RangeError(`hydrographs[${index}].concentration must be finite and nonnegative`);
        const still = segment.still ?? 0;
        if (kind === 'wave' && !Number.isFinite(segment.still)) throw new RangeError(`hydrographs[${index}] (wave) needs a finite still-water level`);
        points += series.length;
        return Object.freeze({ edge, kind, start, end, times, values, concentration, still });
    });
    if (points > MAX_HYDROGRAPH_POINTS) throw new RangeError('hydrograph series exceed 2^20 points in total');
    return Object.freeze(segments);
}

// Boundary face slots (shared with the outflow accumulators): west y, east h + y, south 2h + x,
// north 2h + w + x.
export function boundarySlot(p, edge, k) {
    return { west: k, east: p.height + k, south: 2 * p.height + k, north: 2 * p.height + p.width + k }[edge];
}

// Per-face segment map and segment data in model units. Stage values are shifted by the datum
// (in f64); discharge becomes inflow per unit width over the segment's open (non-solid) cells.
// cumulative[k] is the integral of the series from times[0] to times[k]. speedX and speedY bound
// the ghost wave speeds of the x- and y-normal hydrograph faces over the whole series: a discharge
// ghost at critical depth moves at 3 (g q)^(1/3), a stage ghost at most at 2 sqrt(g h) beyond the
// interior velocity, with h the largest stage above the lowest bed of the segment (`bed` is the
// datum-shifted bed).
export function hydrographSegments(p, datum, solid, bed) {
    const slots = 2 * (p.width + p.height), faceSegment = new Int32Array(slots).fill(-1);
    let speedX = 0, speedY = 0;
    const segments = p.hydrographs.map((s, index) => {
        const vertical = s.edge === 'west' || s.edge === 'east';
        let open = 0, lowest = Infinity;
        for (let k = s.start; k < s.end; k++) {
            const x = vertical ? (s.edge === 'west' ? 0 : p.width - 1) : k;
            const y = vertical ? k : (s.edge === 'south' ? 0 : p.height - 1);
            if (!solid[y * p.width + x]) {
                open++; faceSegment[boundarySlot(p, s.edge, k)] = index;
                lowest = Math.min(lowest, bed[y * p.width + x]);
            }
        }
        if (open === 0) throw new RangeError(`hydrographs[${index}] has no open cells`);
        const kind = { stage: 1, discharge: 2, wave: 3 }[s.kind];
        const scale = kind === 2 ? 1 / (open * (vertical ? p.dy : p.dx)) : 1;
        const values = kind !== 2 ? s.values.map(v => v - datum) : s.values;
        const cumulative = new Float64Array(values.length);
        for (let k = 1; k < values.length; k++) cumulative[k] = cumulative[k - 1] + (s.times[k] - s.times[k - 1]) * (values[k - 1] + values[k]) / 2;
        const peak = Math.max(...values);
        // A wave ghost moves at most at |w| + c <= 3 c_i with c_i from the highest incident level.
        const speed = kind === 2 ? 3 * Math.cbrt(p.gravity * peak * scale) : (kind === 3 ? 3 : 2) * Math.sqrt(p.gravity * Math.max(0, peak - lowest));
        if (vertical) speedX = Math.max(speedX, speed); else speedY = Math.max(speedY, speed);
        return { kind, scale, times: s.times, values, cumulative, concentration: s.concentration, still: s.still - datum };
    });
    return { faceSegment, segments, speedX, speedY };
}

// Timestep bound from the hydrograph ghost speeds over the whole series (seconds; Infinity when
// there are no hydrographs). It plays the role of sourceTimestepLimit for boundary inflow: a dry
// domain whose inflow has not started yet would otherwise take steps limited only by maxDt.
export function hydrographTimestepLimit(p, { speedX, speedY }) {
    const rate = speedX / p.dx + speedY / p.dy;
    return rate > 0 ? p.cfl / rate : Infinity;
}

// Hydrograph clock for the GPU kernels: a two-float f32 sum [hi, lo] of accepted timesteps,
// advanced with TwoSum. Host and kernels perform the same f32 operations, so a step evaluates its
// hydrographs at the same time however steps are batched; hi + lo keeps about 48 bits.
const f32 = Math.fround;
export function clockAdvance([hi, lo], dt) {
    const s = f32(hi + dt), bp = f32(s - hi), err = f32(f32(hi - f32(s - bp)) + f32(dt - bp));
    const l = f32(lo + err), h = f32(s + l);
    return [h, f32(l - f32(h - s))];
}
export const clockTime = ([hi, lo], offset = 0) => f32(hi + f32(lo + offset));

// Linear interpolation with constant extension; binary search (mirrored in the WGSL kernels).
function segmentIndex(times, t) {
    let lo = 0, hi = times.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (times[mid] <= t) lo = mid; else hi = mid; }
    return lo;
}
export function hydrographValue(times, values, t) {
    const n = times.length;
    if (!(t > times[0])) return values[0];
    if (t >= times[n - 1]) return values[n - 1];
    const lo = segmentIndex(times, t), hi = lo + 1;
    return values[lo] + (values[hi] - values[lo]) * ((t - times[lo]) / (times[hi] - times[lo]));
}

// Integral of the series from times[0] to t (constant extension outside the series).
export function hydrographIntegral(times, values, cumulative, t) {
    const n = times.length;
    if (!(t > times[0])) return (t - times[0]) * values[0];
    if (t >= times[n - 1]) return cumulative[n - 1] + (t - times[n - 1]) * values[n - 1];
    const lo = segmentIndex(times, t);
    return cumulative[lo] + (t - times[lo]) * ((values[lo] + hydrographValue(times, values, t)) / 2);
}

// Mean of the series over a step [t0, t1]; discharge faces use it, so a discharge boundary admits
// the integral of Q(t) whatever the timestep. Within one linear piece (or outside the series) the
// mean is the average of the end values, which avoids the cancellation of a difference of
// cumulative integrals in f32; only steps that cross a breakpoint use the cumulative table.
export function hydrographMean(times, values, cumulative, t0, t1) {
    if (!(t1 > t0)) return hydrographValue(times, values, t0);
    const n = times.length;
    const piece = t1 <= times[0] || t0 >= times[n - 1]
        || (t0 > times[0] && t1 < times[n - 1] && segmentIndex(times, t0) === segmentIndex(times, t1));
    if (piece) return (hydrographValue(times, values, t0) + hydrographValue(times, values, t1)) / 2;
    return (hydrographIntegral(times, values, cumulative, t1) - hydrographIntegral(times, values, cumulative, t0)) / (t1 - t0);
}

// GPU boundary buffer of vec2<f32> entries (absolute indices), B = 2(w + h): [0, B) compensated
// water outflow per face slot, [B, 2B) compensated tracer outflow, then this table: [2B, 3B) per
// face (segment + 1, 0); 3B + 3k, +1, +2 per segment k: (kind, scale), (series start, count) and
// (inflow concentration, still-water level of a wave segment); then two entries per point, (t, value)
// and (cumulative integral, 0).
// Integers are stored as exact f32 values.
export function hydrographTable({ faceSegment, segments }) {
    const slots = faceSegment.length, base = 2 * slots;
    const total = slots + 3 * segments.length + segments.reduce((a, s) => a + 2 * s.times.length, 0);
    const table = new Float32Array(2 * total);
    for (let k = 0; k < slots; k++) table[2 * k] = faceSegment[k] + 1;
    let next = base + slots + 3 * segments.length;
    segments.forEach((s, k) => {
        table.set([s.kind, s.scale, next, s.times.length, s.concentration, s.still ?? 0], 2 * (slots + 3 * k));
        for (let j = 0; j < s.times.length; j++) table.set([s.times[j], s.values[j], s.cumulative[j], 0], 2 * (next - base + 2 * j));
        next += 2 * s.times.length;
    });
    if (next >= 1 << 24) throw new RangeError('hydrograph table exceeds the exact f32 index range');
    return table;
}

// Ghost state [h, normal velocity, tangential velocity] for a hydrograph face from the interior
// [h, un, vt] (velocities along +axis and along the other axis); inward = +1 when the interior
// lies on the positive side of the face. Stage: depth = stage - face bed, interior velocity; water
// flowing in comes from rest at the stage (depth loses un²/2g, no tangential velocity); the stage is
// not imposed against supercritical outflow. Discharge: inflow q (m²/s) at depth
// max(h, critical depth), no tangential velocity; zero discharge reflects (wall). Wave: see below.
export function hydrographGhost(kind, value, h, un, vt, zFace, gravity, inward, still = 0) {
    if (kind === 3) {
        // Incident wave (characteristic boundary): the incoming Riemann invariant of a right-going
        // simple wave on the still depth d0 = still - zFace, R = u + 2c = 4 c_i - 2 c_0 with
        // c_i = sqrt(g (d0 + eta)), and the outgoing invariant w - 2c of the interior give the face
        // state; with no incident wave (value = still) waves leave the domain. Written as differences
        // from h so that a lake at the still level gives the interior state exactly.
        const w = inward * un, c = Math.sqrt(gravity * Math.max(h, 0));
        if (w < -c) return [h, un, vt];
        const c0 = Math.sqrt(gravity * Math.max(0, h + ((still - zFace) - h)));
        const ci = Math.sqrt(gravity * Math.max(0, h + ((value - zFace) - h)));
        const delta = (4 * ci - 2 * c0 - w - 2 * c) / 4;
        if (!(c + delta > 0)) return [0, 0, 0];
        const wb = (4 * ci - 2 * c0 + w - 2 * c) / 2;
        return [h + delta * (2 * c + delta) / gravity, inward * wb, wb > 0 ? 0 : vt];
    }
    if (kind === 1) {
        if (inward * un < -Math.sqrt(gravity * h)) return [h, un, vt];
        if (inward * un > 0) return [Math.max(0, value - zFace - un * un / (2 * gravity)), un, 0]; // inflow from rest
        return [Math.max(0, value - zFace), un, vt];
    }
    if (!(value > 0)) return [h, -un, vt];
    const hg = Math.max(h, Math.cbrt(value * value / gravity));
    return [hg, inward * value / hg, 0];
}

// Run-time changes (setParameters, setHydrographs): the options that may change between steps, and a
// configuration with them applied (validated like the original). Hydrograph segments must keep their
// edges and cells, so that the faces they occupy do not change.
export const RUNTIME_OPTIONS = Object.freeze(['wind', 'windDrag', 'coriolis', 'settling']);
export function updatedConfiguration(p, changes) {
    for (const key of Object.keys(changes)) {
        if (!RUNTIME_OPTIONS.includes(key) && key !== 'hydrographs') throw new RangeError(`${key} cannot change during a run`);
    }
    if ('settling' in changes && changes.settling !== null && p.settling === null && !p.tracer) throw new RangeError('settling requires tracer: true');
    const next = configuration({ ...p, ...changes });
    if (changes.hydrographs) {
        const same = next.hydrographs.length === p.hydrographs.length && next.hydrographs.every((s, k) =>
            s.edge === p.hydrographs[k].edge && s.start === p.hydrographs[k].start && s.end === p.hydrographs[k].end && s.kind === p.hydrographs[k].kind);
        if (!same) throw new RangeError('setHydrographs must keep each segment\'s edge, cells and kind');
    }
    return next;
}

// Optional per-cell forcing: Manning n (s/m^(1/3)), depth source rate (m/s; rain or a
// distributed discharge Q*w_i/A_i), the tracer concentration of that source water, and a solid
// (impermeable) mask fixed at creation. Rain carries no tracer.
export function forcingData(p, forcing = {}, ArrayType = Float64Array, solid = undefined) {
    const n = p.width * p.height;
    const read = (key, Type) => {
        const values = forcing[key];
        if (values === undefined) return new Type(n);
        if (values?.length !== n) throw new RangeError(`Expected ${key}.length = width*height`);
        return new Type(values);
    };
    const manning = read('manning', ArrayType), source = read('source', ArrayType);
    const sourceConcentration = read('sourceConcentration', ArrayType);
    solid ??= read('solid', Uint8Array);
    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(manning[i]) || manning[i] < 0) throw new RangeError(`Invalid Manning n at cell ${i}`);
        if (!Number.isFinite(source[i]) || source[i] < 0) throw new RangeError(`Invalid source rate at cell ${i}`);
        if (!Number.isFinite(sourceConcentration[i]) || sourceConcentration[i] < 0) throw new RangeError(`Invalid source concentration at cell ${i}`);
        if (solid[i] > 1) throw new RangeError(`solid must be 0 or 1 at cell ${i}`);
    }
    if (!p.tracer && forcing.sourceConcentration !== undefined) throw new RangeError('sourceConcentration requires tracer: true');
    return { manning, source, sourceConcentration, solid };
}

export function initialData(p, initial, ArrayType = Float64Array) {
    const n = p.width * p.height;
    if (initial.state?.length !== 4 * n || initial.bed?.length !== n) {
        throw new RangeError('Expected state.length = 4*width*height and bed.length = width*height');
    }
    const state = new ArrayType(initial.state);
    const forcing = forcingData(p, initial, ArrayType);
    // Only bed gradients enter the equations. Subtracting a datum in f64 before any
    // rounding keeps f32 eta = h + z precise on high-elevation DEMs ('auto': min open-cell bed).
    let datum = p.datum === 'auto' ? Infinity : p.datum;
    if (p.datum === 'auto') for (let i = 0; i < n; i++) if (!forcing.solid[i]) datum = Math.min(datum, initial.bed[i]);
    if (!Number.isFinite(datum)) datum = 0;
    const bed = ArrayType.from(initial.bed, z => z - datum);
    for (let i = 0; i < n; i++) if (!Number.isFinite(bed[i])) throw new RangeError(`Nonfinite input or negative depth at cell ${i}`);
    const { tracer } = stateData(p, state, forcing.solid, initial.concentration);
    return { state, bed, datum, tracer, ...forcing };
}

// Validates a state [h, hu, hv, 0] per cell in place (finite, h >= 0, zero momentum when dry, solid
// cells dry) and zeroes the fourth (compensation) lane. Tracer runs: concentration c per cell
// (default 0), returned as the conserved amount h c (m).
export function stateData(p, state, solid, concentration = undefined) {
    const n = p.width * p.height;
    if (state.length !== 4 * n) throw new RangeError('Expected state.length = 4*width*height');
    for (let i = 0; i < n; i++) {
        const k = 4 * i;
        if (![state[k], state[k + 1], state[k + 2]].every(Number.isFinite) || state[k] < 0) {
            throw new RangeError(`Nonfinite input or negative depth at cell ${i}`);
        }
        if (state[k] === 0 && (state[k + 1] !== 0 || state[k + 2] !== 0)) {
            throw new RangeError(`Dry cell ${i} has nonzero momentum`);
        }
        if (solid[i] && state[k] !== 0) throw new RangeError(`Solid cell ${i} must be dry`);
        state[k + 3] = 0;
    }
    let tracer = null;
    if (concentration !== undefined && !p.tracer) throw new RangeError('concentration requires tracer: true');
    if (p.tracer) {
        const c = concentration ?? new Float64Array(n);
        if (c.length !== n) throw new RangeError('Expected concentration.length = width*height');
        tracer = new state.constructor(n);
        for (let i = 0; i < n; i++) {
            if (!Number.isFinite(c[i]) || c[i] < 0) throw new RangeError(`Invalid concentration at cell ${i}`);
            tracer[i] = state[4 * i] * c[i];
        }
    }
    return { state, tracer };
}

// Sparse source update (setSources): cells, new rates (m/s, >= 0) and, in tracer runs, optional new
// source concentrations. Returns the validated lists; the caller applies them.
export function sourceUpdates(p, indices, rates, concentrations = undefined) {
    const n = p.width * p.height;
    if (indices.length !== rates.length) throw new RangeError('setSources needs one rate per index');
    if (concentrations !== undefined) {
        if (!p.tracer) throw new RangeError('sourceConcentration requires tracer: true');
        if (concentrations.length !== indices.length) throw new RangeError('setSources needs one concentration per index');
    }
    for (let k = 0; k < indices.length; k++) {
        const i = indices[k];
        if (!Number.isInteger(i) || i < 0 || i >= n) throw new RangeError(`setSources: invalid cell ${i}`);
        if (!Number.isFinite(rates[k]) || rates[k] < 0) throw new RangeError(`Invalid source rate at cell ${i}`);
        if (concentrations !== undefined && (!Number.isFinite(concentrations[k]) || concentrations[k] < 0)) {
            throw new RangeError(`Invalid source concentration at cell ${i}`);
        }
    }
    return { indices, rates, concentrations };
}

// Concentration range [cmin, cmax] of the data: initial concentrations of wet cells, inflow and
// source concentrations, and 0 when rain (tracer-free water) falls. Advection by the conservative
// mass fluxes and diffusion obey a maximum principle within it; cells thinner than H_THIN, where
// h c / h loses precision, have their concentration clipped to it when fluxes are evaluated.
export function tracerBounds(p, { state, tracer, source, sourceConcentration, solid }) {
    let lo = Infinity, hi = -Infinity;
    const take = c => { lo = Math.min(lo, c); hi = Math.max(hi, c); };
    for (let i = 0; i < solid.length; i++) {
        if (solid[i]) continue;
        if (state[4 * i] > 0) take(tracer[i] / state[4 * i]);
        if (source[i] > 0) take(sourceConcentration[i]);
    }
    for (const s of p.hydrographs) take(s.concentration);
    if (p.rain > 0 || p.settling) take(0); // settling lowers concentrations toward 0
    return hi >= lo ? [lo, hi] : [0, 0];
}

// Explicit diffusion bound: D dt (1/dx² + 1/dy²) <= 1/4 keeps the update a convex combination
// together with the advective CFL restriction.
export function diffusionTimestepLimit(p) {
    return p.diffusivity > 0 ? 0.25 / (p.diffusivity * (1 / (p.dx * p.dx) + 1 / (p.dy * p.dy))) : Infinity;
}

// Tracer amount (m³ of concentration-weighted water) entering from per-cell sources, per second.
export function tracerSourceRate(p, { source, sourceConcentration, solid }) {
    let rate = 0;
    for (let i = 0; i < solid.length; i++) if (!solid[i]) rate += source[i] * sourceConcentration[i];
    return rate * p.dx * p.dy;
}

// Concentration of a cell for flux evaluation: h c / h clipped to the data range; 0 when dry.
export function cellConcentration(h, hc, [lo, hi]) {
    return h > 0 ? Math.min(hi, Math.max(lo, hc / h)) : 0;
}

// GPU cell record [z, manning n, source rate, solid] (vec4<f32>); the source concentration
// travels in its own buffer.
export function packCells(bed, { manning, source, solid }) {
    const cells = new Float32Array(4 * bed.length);
    for (let i = 0; i < bed.length; i++) cells.set([bed[i], manning[i], source[i], solid[i]], 4 * i);
    return cells;
}

// Volumetric inflow (m³/s) from uniform rain and per-cell sources; solid cells receive none.
export function sourceDischarge(p, { source, solid }) {
    let rate = 0;
    for (let i = 0; i < solid.length; i++) if (!solid[i]) rate += p.rain + source[i];
    return rate * p.dx * p.dy;
}

// Forcing-aware step bound. On dry or nearly dry ground the wave-speed CFL bound is
// absent, so the depth added in one step, s*dt, must not itself create waves that violate
// CFL: dt * 2 sqrt(g s dt) (1/dx + 1/dy) <= cfl, with s the largest local source (m/s).
export function sourceTimestepLimit(p, forcing) { return sourceLimitFor(p, maxSource(forcing)); }

// Largest source rate (m/s) over open cells.
export function maxSource({ source, solid }) {
    let s = 0;
    for (let i = 0; i < solid.length; i++) if (!solid[i]) s = Math.max(s, source[i]);
    return s;
}

// The bound above for a given largest source rate (uniform rain added).
export function sourceLimitFor(p, largest) {
    const s = largest + p.rain;
    if (!(s > 0)) return Infinity;
    return (p.cfl / (2 * Math.sqrt(p.gravity * s) * (1 / p.dx + 1 / p.dy))) ** (2 / 3);
}

export function validateMaxDt(maxDt) {
    if (!Number.isFinite(maxDt) || maxDt <= 0) throw new RangeError('maxDt must be positive and finite');
}

export function diagnostics(state, p) {
    let volume = 0, minDepth = Infinity, maxDepth = 0, maxSpeed = 0;
    for (let k = 0; k < state.length; k += 4) {
        const h = state[k], qx = state[k + 1], qy = state[k + 2];
        if (![h, qx, qy].every(Number.isFinite) || h < 0) throw new Error(`Invalid state at cell ${k / 4}`);
        volume += h * p.dx * p.dy;
        minDepth = Math.min(minDepth, h);
        maxDepth = Math.max(maxDepth, h);
        maxSpeed = Math.max(maxSpeed, h > 0 ? Math.hypot(qx, qy) / h : 0);
    }
    return { volume, minDepth, maxDepth, maxSpeed };
}

// Low-Froude correction of the normal trace velocities [ul, ur] at an interior face (Thornber et
// al., 2008, with the Froude number for the Mach number): the jump is scaled by z = min(1, Fr), Fr
// the larger of the two trace Froude numbers |u| / sqrt(g h). Unchanged when either side is
// thinner than H_THIN. Mirrored in the MUSCL kernels (lowFroudeVelocities).
export function lowFroudeVelocities(hl, hr, ul, ur, vl, vr, gravity) {
    if (hl <= H_THIN || hr <= H_THIN) return [ul, ur];
    const fr = Math.max(Math.sqrt(ul * ul + vl * vl) / Math.sqrt(gravity * hl), Math.sqrt(ur * ur + vr * vr) / Math.sqrt(gravity * hr));
    const z = Math.min(1, fr), m = 0.5 * (ul + ur), d = 0.5 * (ul - ur);
    return [m + z * d, m - z * d];
}

// Thin-layer depth (m). Stencils containing shallower water reconstruct at first order, and
// momentum in shallower cells is regularized (Kurganov-Petrova velocity desingularization):
// q <- q * sqrt(2) h^2 / sqrt(h^4 + H^4), which equals 1 at h = H and tends to 0 as h -> 0.
export const H_THIN = 1e-4;
export function thinLayerFactor(h) {
    if (!(h < H_THIN)) return 1;
    const h2 = h * h;
    return Math.SQRT2 * h2 / Math.sqrt(h2 * h2 + H_THIN ** 4);
}

// Implicit local Manning update at fixed h: q*r/(r + dt g n² |q|), r = h^(7/3).
// Exact for dq/dt = -g n² |q| q / h^(7/3) over one step; never reverses flow.
export function frictionFactor(h, speedQ, n, gravity, dt) {
    if (!(n > 0) || !(h > 0) || !(speedQ > 0)) return 1;
    const r = h ** (7 / 3), d = r + dt * gravity * n * n * speedQ;
    return d > 0 ? r / d : 1;
}

// Face codes shared by all implementations: 0 interior; -1/1 reflect the left/right
// ghost from the other side (wall or solid cell); -2/2 copy it (open); -4/4 copy if the
// interior flow leaves the domain, otherwise reflect (outflow); 3 closed.
export function faceCode(boundary, first, last, solidLeft, solidRight) {
    let code = 0;
    if ((first || last) && boundary !== 'periodic') code = (first ? -1 : 1) * ({ open: 2, outflow: 4 }[boundary] ?? 1);
    if (solidLeft && solidRight) return 3;
    if (solidLeft) return -1;
    if (solidRight) return 1;
    return code;
}
