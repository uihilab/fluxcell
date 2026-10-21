import { configuration, initialData, forcingData, stateData, sourceUpdates, validateMaxDt, faceCode, frictionFactor, sourceDischarge, sourceTimestepLimit, thinLayerFactor,
    hydrographSegments, hydrographValue, hydrographMean, hydrographGhost, hydrographTimestepLimit,
    tracerBounds, diffusionTimestepLimit, tracerSourceRate, cellConcentration, viscous, localEddyViscosity, viscousSpeed, H_THIN, windStress, settledConcentration, coriolisRotate, updatedConfiguration } from './model.mjs';

// Face record: [mass, x momentum, y momentum, left pressure, right pressure, tracer flux].
const FS = 6;

// Independent f64 reference; no rendering, browser, or GPU dependencies.
export class ReferenceSolver {
    constructor(options, initial) {
        this.config = configuration(options);
        if (this.config.lowFroude && this.constructor === ReferenceSolver) throw new RangeError('lowFroude applies to the MUSCL-Hancock/HLLC schemes');
        Object.assign(this, initialData(this.config, initial));
        this.wind = windStress(this.config); // kinematic wind stress (m²/s²)
        this.next = new Float64Array(this.state.length);
        const { width: w, height: h } = this.config;
        this.facesX = new Float64Array((w + 1) * h * FS);
        this.facesY = new Float64Array(w * (h + 1) * FS);
        this.time = 0;
        // Cumulative volume (m³) through open boundaries (positive outward) and from sources.
        this.boundaryOutflow = 0;
        this.sourceVolume = 0;
        this.stateEditVolume = 0; // volume changed by setState (m³)
        this.sourceLimit = sourceTimestepLimit(this.config, this);
        // Hydrograph faces: [slot, interior cell, axis, inward (+1: interior on the positive side)].
        this.hydro = hydrographSegments(this.config, this.datum, this.solid, this.bed);
        this.hydroLimit = hydrographTimestepLimit(this.config, this.hydro);
        this.hydroFaces = [];
        this.hydro.faceSegment.forEach((segment, slot) => {
            if (segment < 0) return;
            const [cell, axis, inward] = slot < h ? [slot * w, 0, 1] : slot < 2 * h ? [(slot - h) * w + w - 1, 0, -1]
                : slot < 2 * h + w ? [slot - 2 * h, 1, 1] : [(h - 1) * w + slot - 2 * h - w, 1, -1];
            this.hydroFaces.push([slot, cell, axis, inward]);
        });
        this.faceInterval = [0, 0];
        // Passive tracer: conserved amount h c per cell, its outflow through the boundary, and the
        // amount added by sources (inflow through hydrograph faces counts as negative outflow).
        if (this.config.tracer) {
            this.nextTracer = new Float64Array(w * h);
            this.bounds = tracerBounds(this.config, this);
            this.tracerOutflow = 0;
            this.tracerSourceAmount = 0;
            this.tracerDeposited = 0; // amount settled out of the water column (m³ x concentration)
        }
        this.diffusionLimit = diffusionTimestepLimit(this.config);
        // Eddy viscosity per cell, evaluated from the state at the start of each step.
        this.viscous = viscous(this.config);
        if (this.viscous) this.nu = new Float64Array(w * h);
    }

    // Velocity of a (possibly out-of-domain) cell as the GPU tile holds it: mapped like the state
    // (clamped or periodic), the normal component reversed beyond a wall, zero when dry.
    tileVelocity(x, y) {
        const p = this.config;
        const outX = x < 0 || x >= p.width, outY = y < 0 || y >= p.height;
        const mx = p.boundaryX === 'periodic' ? (x + p.width) % p.width : Math.max(0, Math.min(p.width - 1, x));
        const my = p.boundaryY === 'periodic' ? (y + p.height) % p.height : Math.max(0, Math.min(p.height - 1, y));
        const k = 4 * (my * p.width + mx), h = this.state[k];
        if (!(h > 0)) return [0, 0];
        return [this.state[k + 1] / h * (outX && p.boundaryX === 'wall' ? -1 : 1), this.state[k + 2] / h * (outY && p.boundaryY === 'wall' ? -1 : 1)];
    }

    // nu = background + Smagorinsky + bed-generated (model.mjs); zero in solid and thin cells.
    computeEddyViscosity() {
        const p = this.config, m = Math.min(p.dx, p.dy);
        for (let y = 0; y < p.height; y++) for (let x = 0; x < p.width; x++) {
            const i = y * p.width + x, h = this.state[4 * i];
            if (this.solid[i] || !(h > H_THIN)) { this.nu[i] = 0; continue; }
            const [u, v] = this.tileVelocity(x, y);
            let nu = localEddyViscosity(p, h, Math.hypot(u, v), this.manning[i]);
            if (p.smagorinsky > 0) {
                const e = this.tileVelocity(x + 1, y), wv = this.tileVelocity(x - 1, y);
                const n = this.tileVelocity(x, y + 1), s = this.tileVelocity(x, y - 1);
                const ux = (e[0] - wv[0]) / (2 * p.dx), vx = (e[1] - wv[1]) / (2 * p.dx);
                const uy = (n[0] - s[0]) / (2 * p.dy), vy = (n[1] - s[1]) / (2 * p.dy);
                nu += (p.smagorinsky * m) ** 2 * Math.sqrt(2 * ux * ux + 2 * vy * vy + (uy + vx) ** 2);
            }
            this.nu[i] = nu;
        }
    }

    // Viscous momentum flux -h_f nu_f (u_R - u_L) / d across an interior (or periodic) face, with h_f
    // the shallower cell depth; zero when either side is thinner than H_THIN.
    addViscousFlux(left, right, axis, code, output, offset) {
        if (!this.viscous || code !== 0) return;
        const s = this.state, hl = s[4 * left], hr = s[4 * right];
        if (!(hl > H_THIN) || !(hr > H_THIN)) return;
        const hf = Math.min(hl, hr), k = hf * 0.5 * (this.nu[left] + this.nu[right]) / (axis === 0 ? this.config.dx : this.config.dy);
        output[offset + 1] -= k * (s[4 * right + 1] / hr - s[4 * left + 1] / hl);
        output[offset + 2] -= k * (s[4 * right + 2] / hr - s[4 * left + 2] / hl);
    }

    // Concentration of cell i for flux evaluation (clipped to the data range; 0 when dry).
    concentration(i) { return cellConcentration(this.state[4 * i], this.tracer[i], this.bounds); }

    // Concentration on the face side of a cell; face = 0 W, 1 E, 2 S, 3 N. First order here; the
    // MUSCL reference returns its reconstructed trace.
    concentrationTrace(i) { return this.concentration(i); }

    // Advective (upwind with the face mass flux) plus diffusive tracer flux through a face. The
    // ghost side of a domain or solid face copies the interior concentration; water entering
    // through a hydrograph face carries the segment's concentration. Diffusion acts only across
    // interior (and periodic) faces, with the shallower of the two cell depths.
    tracerFlux(left, right, axis, code, slot, mass) {
        if (code === 3) return 0;
        let cl = this.concentrationTrace(left, 2 * axis + 1), cr = this.concentrationTrace(right, 2 * axis);
        const k = slot >= 0 ? this.hydro.faceSegment[slot] : -1;
        if (code < 0) cl = k >= 0 ? this.hydro.segments[k].concentration : cr;
        if (code > 0) cr = k >= 0 ? this.hydro.segments[k].concentration : cl;
        let flux = mass * (mass > 0 ? cl : cr);
        const p = this.config;
        const D = p.diffusivity + (this.viscous ? 0.5 * (this.nu[left] + this.nu[right]) / p.schmidt : 0);
        if (code === 0 && D > 0) {
            const hf = Math.min(this.state[4 * left], this.state[4 * right]);
            flux -= D * hf * (this.concentration(right) - this.concentration(left)) / (axis === 0 ? p.dx : p.dy);
        }
        return flux;
    }

    // [kind (1 stage, 2 discharge, 3 wave), value, still level] of the face's hydrograph at time t;
    // with a step [t, t1], a stage or wave level is taken at the step midpoint (a constant stage stays
    // exact) and a discharge is its mean over the step (its volume is the integral of Q). null for
    // faces without a hydrograph.
    hydroAt(slot, t, t1) {
        const k = slot >= 0 ? this.hydro.faceSegment[slot] : -1;
        if (k < 0) return null;
        const s = this.hydro.segments[k];
        const value = t1 === undefined ? hydrographValue(s.times, s.values, t)
            : s.kind !== 2 ? hydrographValue(s.times, s.values, t + 0.5 * (t1 - t))
            : hydrographMean(s.times, s.values, s.cumulative, t, t1);
        return [s.kind, value * s.scale, s.still];
    }

    // Replace Manning n and/or source rates between steps; the solid mask is fixed.
    // Change wind, wind drag, Coriolis or settling between steps (model.mjs: updatedConfiguration).
    setParameters(changes) {
        this.config = updatedConfiguration(this.config, changes);
        this.wind = windStress(this.config);
        if (this.config.tracer && this.config.settling) this.bounds = [Math.min(0, this.bounds[0]), this.bounds[1]];
    }

    // Replace the hydrograph series (same segments; times on the solver clock) between steps.
    setHydrographs(hydrographs) {
        this.config = updatedConfiguration(this.config, { hydrographs });
        this.hydro = hydrographSegments(this.config, this.datum, this.solid, this.bed);
        this.hydroLimit = hydrographTimestepLimit(this.config, this.hydro);
        if (this.config.tracer) for (const s of this.config.hydrographs) this.bounds = [Math.min(this.bounds[0], s.concentration), Math.max(this.bounds[1], s.concentration)];
    }

    setForcing(forcing) {
        const current = { manning: forcing.manning ?? this.manning, source: forcing.source ?? this.source,
            ...(this.config.tracer ? { sourceConcentration: forcing.sourceConcentration ?? this.sourceConcentration } : {}) };
        Object.assign(this, forcingData(this.config, current, Float64Array, this.solid));
        this.sourceLimit = sourceTimestepLimit(this.config, this);
        if (this.config.tracer) {
            const [lo, hi] = tracerBounds(this.config, this);
            this.bounds = [Math.min(lo, this.bounds[0]), Math.max(hi, this.bounds[1])];
        }
    }

    // Sparse source update between steps: new rates (m/s) and, in tracer runs, optionally new source
    // concentrations for the listed cells (model.mjs: sourceUpdates).
    setSources(indices, rates, concentrations = undefined) {
        sourceUpdates(this.config, indices, rates, concentrations);
        for (let k = 0; k < indices.length; k++) {
            this.source[indices[k]] = rates[k];
            if (concentrations !== undefined) this.sourceConcentration[indices[k]] = concentrations[k];
        }
        this.sourceLimit = sourceTimestepLimit(this.config, this);
        if (this.config.tracer) {
            const [lo, hi] = tracerBounds(this.config, this);
            this.bounds = [Math.min(lo, this.bounds[0]), Math.max(hi, this.bounds[1])];
        }
    }

    // Replace the state [h, hu, hv, 0] per cell between steps (model.mjs: stateData), with the
    // concentration per cell in tracer runs. The volume change (m³) is kept in stateEditVolume, outside
    // the source and boundary budgets.
    setState(state, { concentration } = {}) {
        if (this.config.tracer && concentration === undefined) throw new RangeError('setState needs concentration in tracer runs');
        const next = new Float64Array(state);
        const data = stateData(this.config, next, this.solid, concentration);
        let change = 0;
        for (let i = 0; i < this.solid.length; i++) change += next[4 * i] - this.state[4 * i];
        this.stateEditVolume += change * this.config.dx * this.config.dy;
        this.state = next;
        if (this.config.tracer) {
            this.tracer = data.tracer;
            const [lo, hi] = tracerBounds(this.config, this);
            this.bounds = [Math.min(lo, this.bounds[0]), Math.max(hi, this.bounds[1])];
        }
    }

    // Ghost states for a coded face. Returns [left, right, mass-flux override] with [h, qx, qy, z]
    // states, or null if closed. slot >= 0 marks a domain-boundary face (hydrograph lookup).
    faceStates(left, right, axis, code, slot = -1) {
        const s = this.state, z = this.bed;
        if (code === 3) return null;
        let l = [s[4 * left], s[4 * left + 1], s[4 * left + 2], z[left]];
        let r = [s[4 * right], s[4 * right + 1], s[4 * right + 2], z[right]];
        const hydro = this.hydroAt(slot, ...this.faceInterval);
        if (hydro) {
            const low = code < 0, inside = low ? r : l, inward = low ? 1 : -1, h = inside[0];
            const [hg, un, vt] = hydrographGhost(hydro[0], hydro[1], h, h > 0 ? inside[axis + 1] / h : 0,
                h > 0 ? inside[2 - axis] / h : 0, inside[3], this.config.gravity, inward, hydro[2]);
            const ghost = [hg, 0, 0, inside[3]], override = hydro[0] === 2 ? inward * hydro[1] : undefined;
            ghost[axis + 1] = hg * un; ghost[2 - axis] = hg * vt;
            return low ? [ghost, r, override] : [l, ghost, override];
        }
        if (code === -4) code = r[axis + 1] > 0 ? -1 : -2; // inward at the low side: reflect
        if (code === 4) code = l[axis + 1] < 0 ? 1 : 2;
        if (code < 0) { l = [...r]; if (code === -1) l[axis + 1] = -l[axis + 1]; }
        if (code > 0) { r = [...l]; if (code === 1) r[axis + 1] = -r[axis + 1]; }
        return [l, r, code === 1 || code === -1 ? 0 : undefined]; // reflective faces are exactly impermeable
    }

    face(left, right, axis, code, output, offset, slot = -1) {
        const p = this.config, states = this.faceStates(left, right, axis, code, slot);
        if (!states) { output.fill(0, offset, offset + FS); return; }
        const [[hl, qxl, qyl, zl], [hr, qxr, qyr, zr], override] = states;
        const uLeft = hl > 0 ? qxl / hl : 0, vLeft = hl > 0 ? qyl / hl : 0;
        const ur = hr > 0 ? qxr / hr : 0, vr = hr > 0 ? qyr / hr : 0;
        const top = Math.max(zl, zr);
        const hL = Math.max(0, hl + (zl - top));
        const hR = Math.max(0, hr + (zr - top));
        const nL = axis === 0 ? uLeft : vLeft, nR = axis === 0 ? ur : vr;
        const dryFactor = (hL === 0 || hR === 0) ? 2 : 1;
        const a = Math.max(Math.abs(nL) + dryFactor * Math.sqrt(p.gravity * hL),
            Math.abs(nR) + dryFactor * Math.sqrt(p.gravity * hR));
        const pressureL = 0.5 * p.gravity * hL * hL;
        const pressureR = 0.5 * p.gravity * hR * hR;
        output[offset] = 0.5 * (hL * nL + hR * nR - a * (hR - hL));
        output[offset + 1] = 0.5 * (hL * nL * uLeft + hR * nR * ur
            + (axis === 0 ? pressureL + pressureR : 0) - a * (hR * ur - hL * uLeft));
        output[offset + 2] = 0.5 * (hL * nL * vLeft + hR * nR * vr
            + (axis === 1 ? pressureL + pressureR : 0) - a * (hR * vr - hL * vLeft));
        output[offset + 3] = pressureL;
        output[offset + 4] = pressureR;
        if (override !== undefined) output[offset] = override; // prescribed discharge: exact mass flux
        this.addViscousFlux(left, right, axis, code, output, offset);
        if (p.tracer) output[offset + 5] = this.tracerFlux(left, right, axis, code, slot, output[offset]);
    }

    prepare() {} // Higher-order implementations supply predicted cell traces.
    bedCorrection() { return 0; }

    step(maxDt) {
        validateMaxDt(maxDt);
        const p = this.config, w = p.width, h = p.height, s = this.state, solid = this.solid;
        let ax = 0, ay = 0;
        for (let k = 0; k < s.length; k += 4) {
            const depth = s[k], u = depth > 0 ? s[k + 1] / depth : 0, v = depth > 0 ? s[k + 2] / depth : 0;
            let c = 2 * Math.sqrt(p.gravity * depth);
            // Cell-local eddy viscosity enters the step through an added speed (model.mjs: viscousSpeed).
            if (this.viscous) c += viscousSpeed(p, localEddyViscosity(p, depth, Math.hypot(u, v), this.manning[k / 4]));
            ax = Math.max(ax, Math.abs(u) + c);
            ay = Math.max(ay, Math.abs(v) + c);
        }
        // Hydrograph ghost states (from cell states at the step start) also bound the step.
        for (const [slot, cell, axis, inward] of this.hydroFaces) {
            const [kind, value, still] = this.hydroAt(slot, this.time), k = 4 * cell, depth = s[k];
            const [hg, un] = hydrographGhost(kind, value, depth, depth > 0 ? s[k + 1 + axis] / depth : 0,
                depth > 0 ? s[k + 2 - axis] / depth : 0, this.bed[cell], p.gravity, inward, still);
            const speed = Math.abs(un) + 2 * Math.sqrt(p.gravity * hg);
            if (axis === 0) ax = Math.max(ax, speed); else ay = Math.max(ay, speed);
        }
        const rate = ax / p.dx + ay / p.dy;
        const cap = Math.min(maxDt, this.sourceLimit, this.hydroLimit, this.diffusionLimit);
        const dt = rate > 0 ? Math.min(cap, p.cfl / rate) : cap;
        if (!(dt > 0) || !Number.isFinite(dt)) throw new Error('Invalid timestep');
        this.faceInterval = [this.time, this.time + dt]; // faces use the hydrograph mean over the step
        if (this.viscous) this.computeEddyViscosity();
        this.prepare(dt);
        for (let y = 0; y < h; y++) for (let x = 0; x <= w; x++) {
            let l = y * w + Math.max(0, x - 1), r = y * w + Math.min(x, w - 1);
            if ((x === 0 || x === w) && p.boundaryX === 'periodic') { l = y * w + w - 1; r = y * w; }
            const code = faceCode(p.boundaryX, x === 0, x === w, solid[l], solid[r]);
            const slot = p.boundaryX === 'periodic' ? -1 : x === 0 ? y : x === w ? h + y : -1;
            this.face(l, r, 0, code, this.facesX, FS * (y * (w + 1) + x), slot);
        }
        for (let y = 0; y <= h; y++) for (let x = 0; x < w; x++) {
            let l = Math.max(0, y - 1) * w + x, r = Math.min(y, h - 1) * w + x;
            if ((y === 0 || y === h) && p.boundaryY === 'periodic') { l = (h - 1) * w + x; r = x; }
            const code = faceCode(p.boundaryY, y === 0, y === h, solid[l], solid[r]);
            const slot = p.boundaryY === 'periodic' ? -1 : y === 0 ? 2 * h + x : y === h ? 2 * h + w + x : -1;
            this.face(l, r, 1, code, this.facesY, FS * (y * w + x), slot);
        }
        const fx = this.facesX, fy = this.facesY;
        let deposited = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const i = y * w + x, k = 4 * i, west = FS * (y * (w + 1) + x), east = west + FS;
            const south = FS * (y * w + x), north = south + FS * w;
            if (solid[i]) { this.next.fill(0, k, k + 4); if (p.tracer) this.nextTracer[i] = 0; continue; }
            this.next[k] = s[k] - dt / p.dx * (fx[east] - fx[west])
                - dt / p.dy * (fy[north] - fy[south]) + (p.rain + this.source[i]) * dt;
            // Cell's gh²/2 terms cancel. Subtract each hydrostatic trace pressure
            // before differencing to reduce equilibrium cancellation error.
            this.next[k + 1] = s[k + 1] - dt / p.dx * ((fx[east + 1] - fx[east + 3])
                - (fx[west + 1] - fx[west + 4]) + this.bedCorrection(i, 0)) - dt / p.dy * (fy[north + 1] - fy[south + 1]);
            this.next[k + 2] = s[k + 2] - dt / p.dx * (fx[east + 2] - fx[west + 2])
                - dt / p.dy * ((fy[north + 2] - fy[north + 3]) - (fy[south + 2] - fy[south + 4]) + this.bedCorrection(i, 1));
            if (this.next[k] === 0) { this.next[k + 1] = 0; this.next[k + 2] = 0; }
            // Wind stress on wet cells, before friction (model.mjs: windStress).
            if (this.next[k] > H_THIN) { this.next[k + 1] += dt * this.wind[0]; this.next[k + 2] += dt * this.wind[1]; }
            // Coriolis rotation of the momentum of wet cells (model.mjs: coriolisRotate).
            if (p.coriolis !== 0 && this.next[k] > H_THIN) [this.next[k + 1], this.next[k + 2]] = coriolisRotate(p.coriolis, dt, this.next[k + 1], this.next[k + 2]);
            const factor = frictionFactor(this.next[k], Math.hypot(this.next[k + 1], this.next[k + 2]),
                this.manning[i], p.gravity, dt);
            this.next[k + 1] *= factor; this.next[k + 2] *= factor;
            const thin = thinLayerFactor(this.next[k]);
            this.next[k + 1] *= thin; this.next[k + 2] *= thin;
            if (this.next[k] < 0 || !Number.isFinite(this.next[k] + this.next[k + 1] + this.next[k + 2])) {
                throw new Error(`Rejected invalid update at cell ${i}; state/time unchanged`);
            }
            if (p.tracer) {
                // A cell that dries completely keeps no tracer (its residual is rounding).
                this.nextTracer[i] = this.next[k] === 0 ? 0 : this.tracer[i] - dt / p.dx * (fx[east + 5] - fx[west + 5])
                    - dt / p.dy * (fy[north + 5] - fy[south + 5]) + this.source[i] * this.sourceConcentration[i] * dt;
                if (!Number.isFinite(this.nextTracer[i])) throw new Error(`Rejected invalid update at cell ${i}; state/time unchanged`);
                if (p.settling && this.next[k] > 0 && this.nextTracer[i] > 0) {
                    // Settling at the new depth (model.mjs: settledConcentration); the loss is deposited.
                    const settled = this.next[k] * settledConcentration(p.settling, this.nextTracer[i] / this.next[k], this.next[k], dt);
                    deposited += (this.nextTracer[i] - settled) * p.dx * p.dy;
                    this.nextTracer[i] = settled;
                }
            }
        }
        // Net volume through every non-periodic boundary face (walls contribute only rounding).
        const through = component => {
            let total = 0;
            if (p.boundaryX !== 'periodic') for (let y = 0; y < h; y++) {
                total += (fx[FS * (y * (w + 1) + w) + component] - fx[FS * y * (w + 1) + component]) * dt * p.dy;
            }
            if (p.boundaryY !== 'periodic') for (let x = 0; x < w; x++) {
                total += (fy[FS * (h * w + x) + component] - fy[FS * x + component]) * dt * p.dx;
            }
            return total;
        };
        const outflow = through(0);
        if (p.tracer) {
            this.tracerOutflow += through(5);
            this.tracerSourceAmount += tracerSourceRate(p, this) * dt;
            this.tracerDeposited += deposited;
            [this.tracer, this.nextTracer] = [this.nextTracer, this.tracer];
        }
        [this.state, this.next] = [this.next, this.state];
        this.boundaryOutflow += outflow;
        this.sourceVolume += sourceDischarge(p, this) * dt;
        this.time += dt;
        return dt;
    }

    readState() { return this.state.slice(); }
    // Conserved tracer amount h c per cell (m).
    readTracer() { return this.tracer.slice(); }
}
