import { ReferenceSolver } from './reference.mjs';
import { validateMaxDt, H_THIN, lowFroudeVelocities } from './model.mjs';

export function mcSlope(left, center, right) {
    const a = 2 * (center - left), b = (right - left) / 2, c = 2 * (right - center);
    if (a > 0 && b > 0 && c > 0) return Math.min(a, b, c);
    if (a < 0 && b < 0 && c < 0) return Math.max(a, b, c);
    return 0;
}

// States [h, normal velocity, tangential velocity]. Returns flux in that frame.
export function hllc(left, right, g, onFallback = () => {}) {
    const [hl, ul, vl] = left, [hr, ur, vr] = right;
    if (hl === 0 && hr === 0) return [0, 0, 0];
    const cl = Math.sqrt(g * hl), cr = Math.sqrt(g * hr);
    const pl = g * hl * hl / 2, pr = g * hr * hr / 2;
    const l = [hl, hl * ul, hl * vl], r = [hr, hr * ur, hr * vr];
    const fl = [hl * ul, hl * ul * ul + pl, hl * ul * vl];
    const fr = [hr * ur, hr * ur * ur + pr, hr * ur * vr];
    let sl = Math.min(ul - cl, ur - cr), sr = Math.max(ul + cl, ur + cr);
    if (hl === 0) { sl = ur - 2 * cr; sr = ur + cr; }
    if (hr === 0) { sl = ul - cl; sr = ul + 2 * cl; }
    if (sl >= 0) return fl;
    if (sr <= 0) return fr;
    const fallback = () => {
        onFallback();
        return fl.map((v, k) => (sr * v - sl * fr[k] + sl * sr * (r[k] - l[k])) / (sr - sl));
    };
    if (hl === 0 || hr === 0 || ur - ul >= 2 * (cl + cr)) return fallback();
    const dl = hl * (sl - ul), dr = hr * (sr - ur), denominator = dr - dl;
    if (Math.abs(denominator) <= 1e-6 * Math.max(Math.abs(dl), Math.abs(dr), 1e-30)) return fallback();
    const star = (dr * ur - dl * ul + pl - pr) / denominator;
    const scale = Math.max(Math.abs(sl), Math.abs(sr), 1e-30);
    if (!Number.isFinite(star) || star <= sl + 1e-6 * scale || star >= sr - 1e-6 * scale) return fallback();
    const hsL = dl / (sl - star), hsR = dr / (sr - star);
    if (!(hsL >= 0) || !(hsR >= 0) || !Number.isFinite(hsL + hsR)) return fallback();
    const fromLeft = star >= 0, hs = fromLeft ? hsL : hsR;
    const s = fromLeft ? sl : sr, original = fromLeft ? l : r, f = fromLeft ? fl : fr;
    // Star-region flux F + s (U* - U). Mass and tangential components use the exact identities
    // F*_h = s* h* and F*_v = s* h* v, so a state at rest (s* = 0) has exactly zero mass flux
    // even where f32 division is not correctly rounded (GPU).
    const v = fromLeft ? vl : vr;
    return [star * hs, f[1] + s * (hs * star - original[1]), star * hs * v];
}

// Hydrostatic reconstruction hK* = max(0, etaK - max(zL, zR)) evaluated only from differences.
// Traces t = [eta - eta_c, u, v, z - z_c] of cells with centre depth hc and bed zc. dz is the trace
// bed difference and s the surface difference; the lower side is cut from the higher side's trace
// depth, so a lake at rest (s = 0) yields identical depths on both sides.
export function hydrostaticDepths(hcL, zcL, l, hcR, zcR, r) {
    const hl = hcL + (l[0] - l[3]), hr = hcR + (r[0] - r[3]);
    const dz = (zcL - zcR) + (l[3] - r[3]), s = ((hcL - hcR) + (zcL - zcR)) + (l[0] - r[0]);
    return dz >= 0 ? [hl, Math.min(hr, Math.max(0, hl - s))] : [Math.min(hl, Math.max(0, hr + s)), hr];
}

export class MusclReferenceSolver extends ReferenceSolver {
    constructor(options, initial) {
        super(options, initial);
        // Cell traces W,E,S,N; each stores [eta - eta_c, u, v, z - z_c] relative to the cell's own
        // free surface eta_c and bed z_c. Only differences enter, so f32 implementations keep full
        // depth precision at any elevation and a lake at rest has exactly zero surface slopes.
        this.traces = new Float64Array(this.config.width * this.config.height * 16);
        // Tracer concentration traces W, E, S, N per cell.
        if (this.config.tracer) this.tracerTraces = new Float64Array(this.config.width * this.config.height * 4);
        this.counters = { hlleFallbacks: 0, reconstructionFallbacks: 0, retries: 0 };
    }

    // Mapped cell index for a possibly out-of-domain coordinate.
    cellIndex(x, y) {
        const p = this.config;
        x = p.boundaryX === 'periodic' ? (x + p.width) % p.width : Math.max(0, Math.min(p.width - 1, x));
        y = p.boundaryY === 'periodic' ? (y + p.height) % p.height : Math.max(0, Math.min(p.height - 1, y));
        return y * p.width + x;
    }

    // [(h - hc) + (z - zc), u, v, z - zc, h] for a (possibly ghost) cell relative to the centre (hc, zc).
    primitive(x, y, hc, zc) {
        const p = this.config;
        const outX = x < 0 || x >= p.width, outY = y < 0 || y >= p.height;
        const i = this.cellIndex(x, y), h = this.state[4 * i], dz = this.bed[i] - zc;
        return [(h - hc) + dz, h > 0 ? this.state[4 * i + 1] / h * (outX && p.boundaryX === 'wall' ? -1 : 1) : 0,
            h > 0 ? this.state[4 * i + 2] / h * (outY && p.boundaryY === 'wall' ? -1 : 1) : 0, dz, h];
    }

    // A solid neighbour is the mirror image of the centre across the shared face.
    neighbour(x, y, ox, oy, center, hc, zc) {
        if (!this.solid[this.cellIndex(x + ox, y + oy)]) return this.primitive(x + ox, y + oy, hc, zc);
        const mirror = [...center];
        mirror[ox !== 0 ? 1 : 2] *= -1;
        return mirror;
    }

    concentrationTrace(i, face) { return this.tracerTraces[4 * i + face]; }

    // MC-limited concentration slopes with the Hancock predictor of c_t + u c_x + v c_y = 0, clipped to
    // the range of the cell and its four neighbours (first order where the flow reconstruction is).
    // A solid neighbour mirrors the cell; neighbours outside the domain map as for the flow.
    prepareTracer(i, x, y, u, v, firstOrder, dt) {
        const p = this.config, c = this.concentration(i);
        const nb = (ox, oy) => { const j = this.cellIndex(x + ox, y + oy); return this.solid[j] ? c : this.concentration(j); };
        const cw = nb(-1, 0), ce = nb(1, 0), cs = nb(0, -1), cn = nb(0, 1);
        if (firstOrder || this.solid[i]) { this.tracerTraces.fill(c, 4 * i, 4 * i + 4); return; }
        const sx = mcSlope(cw, c, ce), sy = mcSlope(cs, c, cn);
        const d = -dt / 2 * (u * sx / p.dx + v * sy / p.dy);
        const lo = Math.min(c, cw, ce, cs, cn), hi = Math.max(c, cw, ce, cs, cn);
        [c - sx / 2 + d, c + sx / 2 + d, c - sy / 2 + d, c + sy / 2 + d]
            .forEach((value, face) => { this.tracerTraces[4 * i + face] = Math.min(hi, Math.max(lo, value)); });
    }

    prepare(dt) {
        this.trialDt = dt;
        this.lastCounters = { hlleFallbacks: 0, reconstructionFallbacks: 0 };
        const p = this.config;
        for (let y = 0; y < p.height; y++) for (let x = 0; x < p.width; x++) {
            const i = y * p.width + x, zc = this.bed[i], h = this.state[4 * i];
            const center = this.primitive(x, y, h, zc), base = center.slice(0, 4);
            if (this.solid[i]) {
                for (let face = 0; face < 4; face++) this.traces.set(base, 16 * i + 4 * face);
                if (p.tracer) this.prepareTracer(i, x, y, 0, 0, true, dt);
                continue;
            }
            const west = this.neighbour(x, y, -1, 0, center, h, zc), east = this.neighbour(x, y, 1, 0, center, h, zc);
            const south = this.neighbour(x, y, 0, -1, center, h, zc), north = this.neighbour(x, y, 0, 1, center, h, zc);
            // First order where any stencil depth is thin, or where a bed step to a neighbour exceeds
            // the depth (thin water on cliffs: hydrostatic reconstruction and the predictor are
            // unreliable there).
            let fallback = [center, west, east, south, north].some(v => v[4] <= H_THIN)
                || [west, east, south, north].some(v => Math.abs(v[3]) > h);
            let sx = [0, 1, 2, 3].map(k => fallback ? 0 : mcSlope(west[k], center[k], east[k]));
            let sy = [0, 1, 2, 3].map(k => fallback ? 0 : mcSlope(south[k], center[k], north[k]));
            const depthSlope = Math.max(Math.abs(sx[0] - sx[3]), Math.abs(sy[0] - sy[3]));
            const theta = depthSlope > 0 ? Math.min(1, 2 * h / depthSlope) : 1;
            sx = sx.map(v => v * theta); sy = sy.map(v => v * theta);
            const u = center[1], v = center[2];
            const delta = [
                -dt / 2 * (u * (sx[0] - sx[3]) / p.dx + v * (sy[0] - sy[3]) / p.dy + h * (sx[1] / p.dx + sy[2] / p.dy)),
                -dt / 2 * (u * sx[1] / p.dx + v * sy[1] / p.dy + p.gravity * sx[0] / p.dx),
                -dt / 2 * (u * sx[2] / p.dx + v * sy[2] / p.dy + p.gravity * sy[0] / p.dy), 0,
            ];
            let faces = [[sx, -1], [sx, 1], [sy, -1], [sy, 1]].map(([s, sign]) =>
                base.map((value, k) => value + sign * s[k] / 2 + delta[k]));
            // Trace depth is h + (deta - dz); reject negative or nonfinite traces.
            if (faces.some(f => h + (f[0] - f[3]) < 0 || !f.every(Number.isFinite))) { faces = [base, base, base, base]; fallback = true; }
            if (fallback) this.lastCounters.reconstructionFallbacks++;
            for (let face = 0; face < 4; face++) this.traces.set(faces[face], 16 * i + 4 * face);
            if (p.tracer) this.prepareTracer(i, x, y, u, v, fallback, dt);
        }
    }

    face(left, right, axis, code, output, offset, slot = -1) {
        if (code === 3) { output.fill(0, offset, offset + 6); return; }
        let l = Array.from(this.traces.subarray(16 * left + 4 * (2 * axis + 1), 16 * left + 4 * (2 * axis + 2)));
        let r = Array.from(this.traces.subarray(16 * right + 8 * axis, 16 * right + 8 * axis + 4));
        let zl = this.bed[left], zr = this.bed[right], hl = this.state[4 * left], hr = this.state[4 * right];
        const hydro = this.hydroAt(slot, ...this.faceInterval), g = this.config.gravity;
        let override;
        if (hydro) {
            // Ghost trace relative to the interior cell (hc, zc), sharing the interior face bed.
            const low = code < 0, ti = low ? r : l, hc = low ? hr : hl, zc = low ? zr : zl, inward = low ? 1 : -1;
            const ghost = [...ti], hi = hc + (ti[0] - ti[3]), [kind, value, still] = hydro;
            if (kind === 3) {
                // Incident wave (model.mjs: hydrographGhost) in difference form: the still and
                // incident levels relative to the trace surface; a lake at the still level gives
                // the interior trace exactly.
                const w = inward * ti[axis + 1], c = Math.sqrt(g * Math.max(hi, 0));
                if (!(w < -c)) {
                    const c0 = Math.sqrt(g * Math.max(0, hi + (((still - zc) - hc) - ti[0])));
                    const ci = Math.sqrt(g * Math.max(0, hi + (((value - zc) - hc) - ti[0])));
                    const delta = (4 * ci - 2 * c0 - w - 2 * c) / 4;
                    if (!(c + delta > 0)) { ghost[0] = ti[3] - hc; ghost[axis + 1] = 0; ghost[2 - axis] = 0; }
                    else {
                        const wb = (4 * ci - 2 * c0 + w - 2 * c) / 2;
                        ghost[0] = ti[0] + delta * (2 * c + delta) / g;
                        ghost[axis + 1] = inward * wb;
                        if (wb > 0) ghost[2 - axis] = 0;
                    }
                }
            } else if (kind === 1) {
                // Stage: prescribed surface, interior velocity; not imposed against supercritical outflow.
                if (!(inward * ti[axis + 1] < -Math.sqrt(g * hi))) ghost[0] = Math.max((value - zc) - hc, ti[3] - hc);
                // Inflow (backflow) comes from water at rest at the stage: its depth loses the velocity head
                // un²/2g and it carries no tangential velocity. Copying the interior state feeds boundary
                // eddies back in (unstable on fine grids).
                const un = ti[axis + 1];
                if (inward * un > 0) { ghost[0] = Math.max((value - zc - un * un / (2 * g)) - hc, ti[3] - hc); ghost[2 - axis] = 0; }
            } else if (value > 0) {
                // Discharge: inflow q at depth max(h, critical depth); exact mass flux.
                const hg = Math.max(hi, Math.cbrt(value * value / g));
                ghost[0] = (hg - hc) + ti[3]; ghost[axis + 1] = inward * value / hg; ghost[2 - axis] = 0;
                override = inward * value;
            } else { ghost[axis + 1] *= -1; override = 0; }
            if (low) { l = ghost; zl = zr; hl = hr; } else { r = ghost; zr = zl; hr = hl; }
        } else {
            if (code === -4) code = r[axis + 1] > 0 ? -1 : -2; // inward at the low side: reflect
            if (code === 4) code = l[axis + 1] < 0 ? 1 : 2;
            if (code < 0) { l = [...r]; zl = zr; hl = hr; if (code === -1) l[axis + 1] *= -1; }
            if (code > 0) { r = [...l]; zr = zl; hr = hl; if (code === 1) r[axis + 1] *= -1; }
            if (code === 1 || code === -1) override = 0; // reflective faces are exactly impermeable
        }
        const [hL, hR] = hydrostaticDepths(hl, zl, l, hr, zr, r);
        let [ul, ur] = [l[axis + 1], r[axis + 1]];
        if (code === 0 && this.config.lowFroude) [ul, ur] = lowFroudeVelocities(hL, hR, ul, ur, l[2 - axis], r[2 - axis], g);
        const f = hllc([hL, ul, l[2 - axis]], [hR, ur, r[2 - axis]], this.config.gravity,
            () => this.lastCounters.hlleFallbacks++);
        output[offset] = f[0]; output[offset + 1] = f[axis === 0 ? 1 : 2]; output[offset + 2] = f[axis === 0 ? 2 : 1];
        output[offset + 3] = this.config.gravity * hL * hL / 2;
        output[offset + 4] = this.config.gravity * hR * hR / 2;
        if (override !== undefined) output[offset] = override;
        this.addViscousFlux(left, right, axis, code, output, offset);
        if (this.config.tracer) output[offset + 5] = this.tracerFlux(left, right, axis, code, slot, output[offset]);
    }

    bedCorrection(i, axis) {
        const a = 16 * i + 8 * axis, b = a + 4, t = this.traces, h = this.state[4 * i];
        const hbar = ((h + (t[a] - t[a + 3])) + (h + (t[b] - t[b + 3]))) / 2;
        return this.config.gravity * hbar * (t[b] - t[a]);
    }

    step(maxDt) {
        validateMaxDt(maxDt);
        let cap = maxDt;
        for (let retry = 0; retry <= 8; retry++) {
            try {
                const dt = super.step(cap);
                this.counters.hlleFallbacks += this.lastCounters.hlleFallbacks;
                this.counters.reconstructionFallbacks += this.lastCounters.reconstructionFallbacks;
                return dt;
            } catch (error) {
                if (!error.message.startsWith('Rejected invalid update') || retry === 8) throw error;
                cap = this.trialDt / 2;
                this.counters.retries++;
            }
        }
    }
}
