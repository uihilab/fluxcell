// MUSCL-Hancock / HLLC, fused tiled kernel. Same discretization as
// muscl-hancock.wgsl; reconstruction, face fluxes, update and the next step's wave
// speeds run in one dispatch per step. Each 16x16 workgroup loads a 20x20 tile
// (two-cell halo) into workgroup memory. Faces on workgroup seams are evaluated by
// both neighbours from identical inputs and code, so interior mass fluxes cancel.
// The timestep and remaining time stay on the GPU; the host reads back once per batch.
// With tracer = 1 a passive scalar (conserved amount h c) moves with the same mass fluxes.
struct Params {
    width: u32, height: u32, boundaryX: u32, boundaryY: u32,
    dx: f32, dy: f32, gravity: f32, cfl: f32,
    maxDt: f32, rain: f32, parity: u32, pad: f32,
    tracer: u32, diffusivity: f32, cmin: f32, cmax: f32,
    // Verification only (0 = off): reject, as an invalid update, every step longer than faultDt
    // that starts at or after faultTime, to exercise rollback, prefix replay and retry.
    faultDt: f32, faultTime: f32, lowFroude: u32, pad2: f32,
    // Turbulence closure (model.mjs): background eddy viscosity, Smagorinsky constant, bed
    // coefficient and 1 / turbulent Schmidt number.
    nu0: f32, smagorinsky: f32, bedTurbulence: f32, invSchmidt: f32,
    // Kinematic wind stress tau / rho_w (m²/s²; model.mjs: windStress).
    windX: f32, windY: f32,
    // Tracer settling w_s = settleVelocity * c^settleExponent (m/s; model.mjs: settledConcentration).
    settleVelocity: f32, settleExponent: f32,
    // Coriolis parameter f (1/s; model.mjs: coriolisRotate).
    coriolis: f32, pad5: f32, pad6: f32, pad7: f32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> stateIn: array<vec4<f32>>;
// [bed z (m), Manning n, source rate (m/s), solid (0/1)].
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> stateOut: array<vec4<f32>>;
// 0 dt; 1,2 speed maxima slot 0; 3,4 slot 1; 5 error bits; 6 HLL fallbacks;
// 7 reconstruction fallbacks; 8 step index; 9 remaining time; 10 first failing step;
// 11, 12 hydrograph clock (hi, lo) at the current step's start; 13, 14 running clock; 16.. dt log.
@group(0) @binding(4) var<storage, read_write> control: array<atomic<u32>>;
// Compensated open-boundary outflow volume per boundary face slot (sum, compensation), then the
// tracer outflow per slot, then the hydrograph table.
@group(0) @binding(5) var<storage, read_write> boundary: array<vec2<f32>>;
// Tracer amount h c with its compensation (ping-pong pair) and the concentration of source water.
// Declared read_write to share the runner's bind group layout; tracerIn and tracerSource are only read.
@group(0) @binding(6) var<storage, read_write> tracerIn: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> tracerOut: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> tracerSource: array<f32>;

// Thin-layer depth (m); see thinLayerFactor in model.mjs.
const H_THIN = 1e-4;
const H_THIN4 = 1e-16;
// cos and sin of a rotation angle. WGSL guarantees sin and cos only to an absolute 2^-11, far too
// coarse for Coriolis angles f dt ~ 1e-5. The angle is halved until it is below 0.1 rad, where the
// Taylor series (error < b^7/5040) is accurate to f32 rounding, and the rotation is doubled back.
fn rotation(a: f32) -> vec2<f32> {
    var b = a; var k = 0;
    while abs(b) >= 0.1 && k < 30 { b *= 0.5; k++; }
    let b2 = b * b;
    var cs = vec2<f32>(1.0 - b2 * (0.5 - b2 * (1.0 / 24.0 - b2 / 720.0)), b * (1.0 - b2 * (1.0 / 6.0 - b2 * (1.0 / 120.0 - b2 / 5040.0))));
    for (var j = 0; j < k; j++) { cs = vec2<f32>(cs.x * cs.x - cs.y * cs.y, 2.0 * cs.x * cs.y); }
    return cs;
}
const LOG = 16u;
const MAX_LOG = 256u;
const TX = 16u;
const TY = 16u;
const PX = 20u;
const PY = 20u;
// Phase 1: [h,u,v,z] tile (PX*PY); bed and solid flags are also kept in their own arrays. Phase 2: E traces [0,256), N traces [256,512),
// halo traces [512,576) (E of column -1, W of column TX, N of row -1, S of row TY).
// Phase 3: west/south face records [flux.xyz, left-side hydrostatic pressure].
var<workgroup> tile: array<vec4<f32>, 576>;
var<workgroup> solidTile: array<u32, 400>;
var<workgroup> bedTile: array<f32, 400>;
var<workgroup> depthTile: array<f32, 400>;
// Tracer: cell concentrations (phase 1), then concentration traces in the layout of `tile`
// (phase 2), then west/south tracer face fluxes (phase 3). Workgroup memory totals 16368 bytes.
var<workgroup> cTile: array<f32, 576>;
var<workgroup> wgMax: array<atomic<u32>, 2>;
var<workgroup> wgCount: array<atomic<u32>, 3>;
var<workgroup> wgTime: vec4<f32>; // dt, hydrograph clock (hi, lo)

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn velocity(u: vec4<f32>) -> vec2<f32> {
    if u.x > 0.0 { return u.yz / u.x; }
    return vec2<f32>(0.0);
}
fn speeds(u: vec4<f32>) -> vec2<f32> { return abs(velocity(u)) + vec2<f32>(2.0 * sqrt(p.gravity * u.x)); }
fn viscous() -> bool { return p.nu0 > 0.0 || p.smagorinsky > 0.0 || p.bedTurbulence > 0.0; }
// Cell-local eddy viscosity (background + bed-generated) and the speed it adds to the timestep
// reduction (model.mjs: localEddyViscosity, viscousSpeed).
fn localNu(h: f32, speed: f32, n: f32) -> f32 {
    if !(h > H_THIN) { return 0.0; }
    var nu = p.nu0;
    if p.bedTurbulence > 0.0 && n > 0.0 { nu += p.bedTurbulence * sqrt(p.gravity) * n * speed * pow(h, 5.0 / 6.0); }
    return nu;
}
fn viscousSpeed(u: vec4<f32>, n: f32) -> f32 {
    if !viscous() { return 0.0; }
    var nu = localNu(u.x, length(velocity(u)), n);
    if p.tracer != 0u { nu *= max(1.0, p.invSchmidt); }
    return 4.0 * p.cfl * nu * (1.0 / (p.dx * p.dx) + 1.0 / (p.dy * p.dy)) / (1.0 / p.dx + 1.0 / p.dy);
}

// Hydrograph clock: a two-float f32 sum (hi, lo) of accepted timesteps, advanced with TwoSum.
// The host repeats the same f32 operations (model.mjs: clockAdvance), so every step evaluates its
// hydrographs at the same time however the steps are batched.
fn clockAdvance(c: vec2<f32>, dt: f32) -> vec2<f32> {
    let s = c.x + dt; let bp = s - c.x; let err = (c.x - (s - bp)) + (dt - bp);
    let l = c.y + err; let h = s + l;
    return vec2<f32>(h, l - (h - s));
}
fn clockTime(c: vec2<f32>, offset: f32) -> f32 { return c.x + (c.y + offset); }
// Hydrograph boundaries (model.mjs: hydrographTable, hydrographValue, hydrographGhost).
// Boundary face slots: west y, east h + y, south 2h + x, north 2h + w + x. The table follows
// the water and tracer outflow accumulators, 2(w + h) each, in the boundary buffer.
const NO_SLOT = 0xffffffffu;
// Series of a face: kind (0 none, 1 stage, 2 discharge, 3 wave), scale, first point entry, point
// count; still: still-water level of a wave segment.
// Each point occupies two entries: (t, value) and (cumulative integral, 0).
// conc: concentration of the water entering through the segment.
struct Series { kind: f32, scale: f32, start: u32, count: u32, conc: f32, still: f32 }
fn seriesOf(slot: u32) -> Series {
    if slot == NO_SLOT { return Series(0.0, 0.0, 0u, 0u, 0.0, 0.0); }
    let b = 2u * (p.width + p.height);
    let segment = u32(boundary[2u * b + slot].x);
    if segment == 0u { return Series(0.0, 0.0, 0u, 0u, 0.0, 0.0); }
    let head = boundary[3u * b + 3u * (segment - 1u)];
    let span = boundary[3u * b + 3u * (segment - 1u) + 1u];
    let extra = boundary[3u * b + 3u * (segment - 1u) + 2u];
    return Series(head.x, head.y, u32(span.x), u32(span.y), extra.x, extra.y);
}
fn seriesPoint(s: Series, j: u32) -> vec2<f32> { return boundary[s.start + 2u * j]; }
// Index of the point that starts the segment containing t (times[0] < t < times[last]).
fn seriesSegment(s: Series, t: f32) -> u32 {
    var lo = 0u; var hi = s.count - 1u;
    loop {
        if hi - lo <= 1u { break; }
        let mid = (lo + hi) / 2u;
        if seriesPoint(s, mid).x <= t { lo = mid; } else { hi = mid; }
    }
    return lo;
}
fn seriesValue(s: Series, t: f32) -> f32 {
    let first = seriesPoint(s, 0u); let last = seriesPoint(s, s.count - 1u);
    if !(t > first.x) { return first.y; }
    if t >= last.x { return last.y; }
    let lo = seriesSegment(s, t); let a = seriesPoint(s, lo); let c = seriesPoint(s, lo + 1u);
    return a.y + (c.y - a.y) * ((t - a.x) / (c.x - a.x));
}
fn seriesIntegral(s: Series, t: f32) -> f32 {
    let first = seriesPoint(s, 0u); let last = seriesPoint(s, s.count - 1u);
    if !(t > first.x) { return (t - first.x) * first.y; }
    if t >= last.x { return boundary[s.start + 2u * (s.count - 1u) + 1u].x + (t - last.x) * last.y; }
    let lo = seriesSegment(s, t); let a = seriesPoint(s, lo);
    return boundary[s.start + 2u * lo + 1u].x + (t - a.x) * ((a.y + seriesValue(s, t)) / 2.0);
}
// (kind, value) of a face's hydrograph at time t (ghost wave speeds).
fn hydrograph(slot: u32, t: f32) -> vec3<f32> {
    let s = seriesOf(slot);
    if s.kind == 0.0 { return vec3<f32>(0.0); }
    return vec3<f32>(s.kind, seriesValue(s, t) * s.scale, s.still);
}
// (kind, value) for the face flux over the step [t0, t1] (model.mjs: hydrographMean, hydroAt): a
// stage at the step midpoint, so a constant stage stays exact; a discharge as its mean over the step,
// so the boundary admits the integral of Q(t). Within one linear piece the mean is the average of the
// end values; only steps that cross a breakpoint use the cumulative table (f32 cancellation).
fn hydrographMean(slot: u32, t0: f32, t1: f32) -> vec3<f32> {
    let s = seriesOf(slot);
    if s.kind == 0.0 { return vec3<f32>(0.0); }
    if s.kind == 1.0 || s.kind == 3.0 { return vec3<f32>(s.kind, seriesValue(s, t0 + 0.5 * (t1 - t0)) * s.scale, s.still); }
    if !(t1 > t0) { return vec3<f32>(s.kind, seriesValue(s, t0) * s.scale, s.still); }
    let first = seriesPoint(s, 0u).x; let last = seriesPoint(s, s.count - 1u).x;
    var piece = t1 <= first || t0 >= last;
    if !piece && t0 > first && t1 < last { piece = seriesSegment(s, t0) == seriesSegment(s, t1); }
    if piece { return vec3<f32>(s.kind, 0.5 * (seriesValue(s, t0) + seriesValue(s, t1)) * s.scale, s.still); }
    return vec3<f32>(s.kind, (seriesIntegral(s, t1) - seriesIntegral(s, t0)) / (t1 - t0) * s.scale, s.still);
}
// Ghost [h, normal velocity, tangential velocity] from the interior [h, un, vt]; inward = +1 when
// the interior lies on the positive side of the face. Kind 1 stage, 2 discharge, 3 wave (still:
// still-water level).
fn hydrographGhost(kind: f32, value: f32, h: f32, un: f32, vt: f32, zFace: f32, inward: f32, still: f32) -> vec3<f32> {
    if kind == 3.0 {
        // Incident wave (characteristic boundary, model.mjs): incoming invariant 4 c_i - 2 c_0 from
        // the incident level, outgoing invariant w - 2c from the interior; with no incident wave the
        // boundary lets waves leave. Differences relative to h keep a lake at the still level exact.
        let w = inward * un; let c = sqrt(p.gravity * max(h, 0.0));
        if w < -c { return vec3<f32>(h, un, vt); }
        let c0 = sqrt(p.gravity * max(0.0, h + ((still - zFace) - h)));
        let ci = sqrt(p.gravity * max(0.0, h + ((value - zFace) - h)));
        let delta = (4.0 * ci - 2.0 * c0 - w - 2.0 * c) / 4.0;
        if !(c + delta > 0.0) { return vec3<f32>(0.0); }
        let wb = (4.0 * ci - 2.0 * c0 + w - 2.0 * c) / 2.0;
        return vec3<f32>(h + delta * (2.0 * c + delta) / p.gravity, inward * wb, select(vt, 0.0, wb > 0.0));
    }
    if kind == 1.0 {
        if inward * un < -sqrt(p.gravity * h) { return vec3<f32>(h, un, vt); }
        // Inflow from water at rest at the stage: velocity head lost, no tangential velocity.
        if inward * un > 0.0 { return vec3<f32>(max(0.0, value - zFace - un * un / (2.0 * p.gravity)), un, 0.0); }
        return vec3<f32>(max(0.0, value - zFace), un, vt);
    }
    if !(value > 0.0) { return vec3<f32>(h, -un, vt); }
    let hg = max(h, pow(value * value / p.gravity, 1.0 / 3.0));
    return vec3<f32>(hg, inward * value / hg, 0.0);
}
fn ghostSpeed(slot: u32, t: f32, u: vec4<f32>, zc: f32, axis: u32, inward: f32) -> f32 {
    let hv = hydrograph(slot, t);
    if hv.x == 0.0 { return 0.0; }
    var v = vec2<f32>(0.0);
    if u.x > 0.0 { v = u.yz / u.x; }
    let g = hydrographGhost(hv.x, hv.y, u.x, v[axis], v[1u - axis], zc, inward, hv.z);
    return abs(g.y) + 2.0 * sqrt(p.gravity * g.x);
}
// Hydrograph ghost speeds (x, y) on the boundary faces of cell (x, y); they bound the timestep.
fn boundarySpeeds(x: u32, y: u32, u: vec4<f32>, zc: f32, t: f32) -> vec2<f32> {
    var s = vec2<f32>(0.0);
    let w = p.width; let h = p.height;
    if p.boundaryX != 1u {
        if x == 0u { s.x = max(s.x, ghostSpeed(y, t, u, zc, 0u, 1.0)); }
        if x == w - 1u { s.x = max(s.x, ghostSpeed(h + y, t, u, zc, 0u, -1.0)); }
    }
    if p.boundaryY != 1u {
        if y == 0u { s.y = max(s.y, ghostSpeed(2u * h + x, t, u, zc, 1u, 1.0)); }
        if y == h - 1u { s.y = max(s.y, ghostSpeed(2u * h + w + x, t, u, zc, 1u, -1.0)); }
    }
    return s;
}
// Slot of the x face west of column x (x == w: the east edge) and of the y face south of row y.
fn slotX(x: u32, y: u32) -> u32 {
    if p.boundaryX == 1u { return NO_SLOT; }
    if x == 0u { return y; }
    if x == p.width { return p.height + y; }
    return NO_SLOT;
}
fn slotY(x: u32, y: u32) -> u32 {
    if p.boundaryY == 1u { return NO_SLOT; }
    if y == 0u { return 2u * p.height + x; }
    if y == p.height { return 2u * p.height + p.width + x; }
    return NO_SLOT;
}
// Hydrograph ghost trace relative to the interior cell (hc, zc), sharing its face bed. Returns the
// trace and the mass flux that replaces the Riemann one when prescribed is set.
struct GhostTrace { trace: vec4<f32>, mass: f32, prescribed: bool }
fn ghostTrace(hv: vec3<f32>, ti: vec4<f32>, hc: f32, zc: f32, axis: u32, inward: f32) -> GhostTrace {
    var g = ti; let hi = hc + (ti.x - ti.w);
    if hv.x == 3.0 {
        // Incident wave (characteristic boundary) in difference form: the still and incident levels
        // relative to the trace surface, so a lake at the still level gives the interior trace exactly.
        let w = inward * ti[axis + 1u]; let c = sqrt(p.gravity * max(hi, 0.0));
        if w < -c { return GhostTrace(g, 0.0, false); }
        let c0 = sqrt(p.gravity * max(0.0, hi + (((hv.z - zc) - hc) - ti.x)));
        let ci = sqrt(p.gravity * max(0.0, hi + (((hv.y - zc) - hc) - ti.x)));
        let delta = (4.0 * ci - 2.0 * c0 - w - 2.0 * c) / 4.0;
        if !(c + delta > 0.0) { g.x = ti.w - hc; g[axis + 1u] = 0.0; g[2u - axis] = 0.0; return GhostTrace(g, 0.0, false); }
        let wb = (4.0 * ci - 2.0 * c0 + w - 2.0 * c) / 2.0;
        g.x = ti.x + delta * (2.0 * c + delta) / p.gravity;
        g[axis + 1u] = inward * wb;
        if wb > 0.0 { g[2u - axis] = 0.0; }
        return GhostTrace(g, 0.0, false);
    }
    if hv.x == 1.0 {
        // Stage: prescribed surface, interior velocity; not imposed against supercritical outflow.
        if !(inward * ti[axis + 1u] < -sqrt(p.gravity * hi)) { g.x = max((hv.y - zc) - hc, ti.w - hc); }
        // Inflow (backflow) comes from water at rest at the stage: its depth loses the velocity head
        // un²/2g and it carries no tangential velocity. Copying the interior state feeds boundary
        // eddies back in (unstable on fine grids).
        let un = ti[axis + 1u];
        if inward * un > 0.0 { g.x = max((hv.y - zc - un * un / (2.0 * p.gravity)) - hc, ti.w - hc); g[2u - axis] = 0.0; }
        return GhostTrace(g, 0.0, false);
    }
    if hv.y > 0.0 {
        // Discharge: inflow q at depth max(h, critical depth); exact mass flux.
        let hg = max(hi, pow(hv.y * hv.y / p.gravity, 1.0 / 3.0));
        g.x = (hg - hc) + ti.w; g[axis + 1u] = inward * hv.y / hg; g[2u - axis] = 0.0;
        return GhostTrace(g, inward * hv.y, true);
    }
    g[axis + 1u] = -g[axis + 1u];
    return GhostTrace(g, 0.0, true);
}

@compute @workgroup_size(256)
fn waveSpeeds(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>,
              @builtin(local_invocation_index) li: u32) {
    if li == 0u { atomicStore(&wgMax[0], 0u); atomicStore(&wgMax[1], 0u); atomicStore(&wgCount[2], 0u); }
    workgroupBarrier();
    let i = id.x + id.y * groups.x * 256u;
    if i < p.width * p.height {
        let u = stateIn[i];
        let clock = vec2<f32>(bitcast<f32>(atomicLoad(&control[13])), bitcast<f32>(atomicLoad(&control[14])));
        let speed = max(speeds(u) + vec2<f32>(viscousSpeed(u, cells[i].y)), boundarySpeeds(i % p.width, i / p.width, u, cells[i].x, clockTime(clock, 0.0)));
        if !finite(speed.x) || !finite(speed.y) || u.x < 0.0 { atomicOr(&wgCount[2], 1u); }
        else { atomicMax(&wgMax[0], bitcast<u32>(speed.x)); atomicMax(&wgMax[1], bitcast<u32>(speed.y)); }
    }
    workgroupBarrier();
    if li == 0u {
        let slot = 1u + 2u * p.parity;
        atomicMax(&control[slot], atomicLoad(&wgMax[0]));
        atomicMax(&control[slot + 1u], atomicLoad(&wgMax[1]));
        let errors = atomicLoad(&wgCount[2]);
        if errors != 0u { atomicOr(&control[5], errors); atomicMin(&control[10], 0u); }
    }
}

@compute @workgroup_size(1)
fn timestep() {
    let slot = 1u + 2u * p.parity; let other = 3u - 2u * p.parity;
    let rate = bitcast<f32>(atomicLoad(&control[slot])) / p.dx + bitcast<f32>(atomicLoad(&control[slot + 1u])) / p.dy;
    let remaining = bitcast<f32>(atomicLoad(&control[9]));
    var dt = min(p.maxDt, remaining);
    if rate > 0.0 { dt = min(dt, p.cfl / rate); }
    let index = atomicAdd(&control[8], 1u);
    if !finite(rate) || !finite(dt) || dt < 0.0 || (dt == 0.0 && remaining > 0.0) {
        atomicOr(&control[5], 2u); atomicMin(&control[10], index);
    }
    atomicStore(&control[0], bitcast<u32>(dt));
    atomicStore(&control[9], bitcast<u32>(remaining - dt));
    let clock = vec2<f32>(bitcast<f32>(atomicLoad(&control[13])), bitcast<f32>(atomicLoad(&control[14])));
    let next = clockAdvance(clock, dt);
    atomicStore(&control[11], bitcast<u32>(clock.x)); atomicStore(&control[12], bitcast<u32>(clock.y));
    atomicStore(&control[13], bitcast<u32>(next.x)); atomicStore(&control[14], bitcast<u32>(next.y));
    if index < MAX_LOG { atomicStore(&control[LOG + index], bitcast<u32>(dt)); }
    atomicStore(&control[other], 0u);
    atomicStore(&control[other + 1u], 0u);
}

fn mapX(x: i32) -> i32 {
    let w = i32(p.width);
    if p.boundaryX == 1u { return ((x % w) + w) % w; }
    return clamp(x, 0, w - 1);
}
fn mapY(y: i32) -> i32 {
    let h = i32(p.height);
    if p.boundaryY == 1u { return ((y % h) + h) % h; }
    return clamp(y, 0, h - 1);
}

// Tile entry as the [(h - hc) + (z - zc), u, v, z - zc] primitive of the multi-pass reconstruction.
fn prim(k: u32, hc: f32, zc: f32) -> vec4<f32> { let t = tile[k]; let dz = t.w - zc; return vec4<f32>((t.x - hc) + dz, t.y, t.z, dz); }
fn depth(k: u32, hc: f32) -> f32 { if solidTile[k] != 0u { return hc; } return tile[k].x; }
fn neighbour(k: u32, c: vec4<f32>, axis: u32, hc: f32, zc: f32) -> vec4<f32> {
    if solidTile[k] == 0u { return prim(k, hc, zc); }
    var m = c;
    m[axis + 1u] = -m[axis + 1u];
    return m;
}

fn mc(a: f32, b: f32, c: f32) -> f32 {
    let dl = 2.0 * (b - a); let dc = 0.5 * (c - a); let dr = 2.0 * (c - b);
    if dl > 0.0 && dc > 0.0 && dr > 0.0 { return min(dl, min(dc, dr)); }
    if dl < 0.0 && dc < 0.0 && dr < 0.0 { return max(dl, max(dc, dr)); }
    return 0.0;
}

struct Traces { w: vec4<f32>, e: vec4<f32>, s: vec4<f32>, n: vec4<f32>, fallback: bool }

fn reconstruct(tx: u32, ty: u32, halfDt: f32) -> Traces {
    let k = ty * PX + tx; let zc = tile[k].w; let h = tile[k].x; let c = prim(k, h, zc);
    if solidTile[k] != 0u { return Traces(c, c, c, c, false); }
    let w = neighbour(k - 1u, c, 0u, h, zc); let e = neighbour(k + 1u, c, 0u, h, zc);
    let s = neighbour(k - PX, c, 1u, h, zc); let n = neighbour(k + PX, c, 1u, h, zc);
    var fallback = min(h, min(min(depth(k - 1u, h), depth(k + 1u, h)), min(depth(k - PX, h), depth(k + PX, h)))) <= H_THIN;
    // Bed steps larger than the depth: hydrostatic reconstruction and the Hancock predictor are
    // unreliable there (thin water on cliffs); reconstruct at first order.
    fallback = fallback || max(max(abs(w.w), abs(e.w)), max(abs(s.w), abs(n.w))) > h;
    var sx = vec4<f32>(0.0); var sy = vec4<f32>(0.0);
    if !fallback {
        for (var j = 0u; j < 4u; j++) { sx[j] = mc(w[j], c[j], e[j]); sy[j] = mc(s[j], c[j], n[j]); }
    }
    let slope = max(abs(sx.x - sx.w), abs(sy.x - sy.w));
    var theta = 1.0;
    if slope > 0.0 { theta = min(1.0, 2.0 * h / slope); }
    sx *= theta; sy *= theta;
    let delta = -halfDt * vec4<f32>(
        c.y * (sx.x - sx.w) / p.dx + c.z * (sy.x - sy.w) / p.dy + h * (sx.y / p.dx + sy.z / p.dy),
        c.y * sx.y / p.dx + c.z * sy.y / p.dy + p.gravity * sx.x / p.dx,
        c.y * sx.z / p.dx + c.z * sy.z / p.dy + p.gravity * sy.x / p.dy, 0.0);
    var faces = array<vec4<f32>, 4>(c - 0.5 * sx + delta, c + 0.5 * sx + delta,
        c - 0.5 * sy + delta, c + 0.5 * sy + delta);
    var invalid = false;
    for (var j = 0u; j < 4u; j++) {
        let f = faces[j];
        invalid = invalid || h + (f.x - f.w) < 0.0 || !finite(f.x) || !finite(f.y) || !finite(f.z) || !finite(f.w);
    }
    if invalid { return Traces(c, c, c, c, true); }
    return Traces(faces[0], faces[1], faces[2], faces[3], fallback);
}

struct Riemann { f: vec3<f32>, fallback: bool }

fn hlle(fl: vec3<f32>, fr: vec3<f32>, l: vec3<f32>, r: vec3<f32>, sl: f32, sr: f32) -> Riemann {
    return Riemann((sr * fl - sl * fr + sl * sr * (r - l)) / (sr - sl), true);
}

fn hllc(lv: vec3<f32>, rv: vec3<f32>) -> Riemann {
    let hl = lv.x; let hr = rv.x; let ul = lv.y; let ur = rv.y;
    if hl == 0.0 && hr == 0.0 { return Riemann(vec3<f32>(0.0), false); }
    let cl = sqrt(p.gravity * hl); let cr = sqrt(p.gravity * hr);
    let pl = 0.5 * p.gravity * hl * hl; let pr = 0.5 * p.gravity * hr * hr;
    let l = vec3<f32>(hl, hl * lv.yz); let r = vec3<f32>(hr, hr * rv.yz);
    let fl = vec3<f32>(hl * ul, hl * ul * ul + pl, hl * ul * lv.z);
    let fr = vec3<f32>(hr * ur, hr * ur * ur + pr, hr * ur * rv.z);
    var sl = min(ul - cl, ur - cr); var sr = max(ul + cl, ur + cr);
    if hl == 0.0 { sl = ur - 2.0 * cr; sr = ur + cr; }
    if hr == 0.0 { sl = ul - cl; sr = ul + 2.0 * cl; }
    if sl >= 0.0 { return Riemann(fl, false); }
    if sr <= 0.0 { return Riemann(fr, false); }
    if hl == 0.0 || hr == 0.0 || ur - ul >= 2.0 * (cl + cr) { return hlle(fl, fr, l, r, sl, sr); }
    let dl = hl * (sl - ul); let dr = hr * (sr - ur); let d = dr - dl;
    if abs(d) <= 1e-6 * max(max(abs(dl), abs(dr)), 1e-30) { return hlle(fl, fr, l, r, sl, sr); }
    let star = (dr * ur - dl * ul + pl - pr) / d;
    let scale = max(max(abs(sl), abs(sr)), 1e-30);
    if !finite(star) || star <= sl + 1e-6 * scale || star >= sr - 1e-6 * scale { return hlle(fl, fr, l, r, sl, sr); }
    let hsL = dl / (sl - star); let hsR = dr / (sr - star);
    if hsL < 0.0 || hsR < 0.0 || !finite(hsL) || !finite(hsR) { return hlle(fl, fr, l, r, sl, sr); }
    // Star-region flux F + s (U* - U); mass and tangential components use F*_h = s* h* and
    // F*_v = s* h* v, so a state at rest (s* = 0) has exactly zero mass flux although GPU f32
    // division is not correctly rounded.
    var fstar = vec3<f32>(star * hsR, fr.y + sr * (hsR * star - r.y), star * hsR * rv.z);
    if star >= 0.0 { fstar = vec3<f32>(star * hsL, fl.y + sl * (hsL * star - l.y), star * hsL * lv.z); }
    return Riemann(fstar, false);
}

// Hydrostatic reconstruction hK* = max(0, etaK - max(zL, zR)) evaluated only from differences.
// Traces are [eta - eta_c, u, v, z - z_c]; the lower side is cut from the higher side's trace
// depth using the surface difference s, so a lake at rest (s = 0) gives identical depths.
fn hydrostaticDepths(hcL: f32, zcL: f32, l: vec4<f32>, hcR: f32, zcR: f32, r: vec4<f32>) -> vec2<f32> {
    let hl = hcL + (l.x - l.w); let hr = hcR + (r.x - r.w);
    let dz = (zcL - zcR) + (l.w - r.w); let s = ((hcL - hcR) + (zcL - zcR)) + (l.x - r.x);
    if dz >= 0.0 { return vec2<f32>(hl, min(hr, max(0.0, hl - s))); }
    return vec2<f32>(min(hl, max(0.0, hr + s)), hr);
}

// Face codes: 0 interior; -1/1 reflect left/right ghost; -2/2 copy it (open);
// -4/4 copy if the interior flow leaves, else reflect (outflow); 3 closed.
fn faceCode(boundaryType: u32, first: bool, last: bool, sl: bool, sr: bool) -> i32 {
    var code = 0;
    if (first || last) && boundaryType != 1u {
        var kind = 1;
        if boundaryType == 2u { kind = 2; }
        if boundaryType == 3u { kind = 4; }
        code = select(1, -1, first) * kind;
    }
    if sl && sr { return 3; }
    if sl { return -1; }
    if sr { return 1; }
    return code;
}

struct FaceFlux { flux: vec3<f32>, pl: f32, pr: f32, fallback: bool }

// Low-Froude correction (Thornber et al., 2008, with the Froude number in place of the Mach number):
// on interior faces the jump in normal velocity entering the Riemann solver is scaled by
// z = min(1, Fr), Fr the larger trace Froude number, so the dissipation of slow vortical flow no
// longer scales with the gravity-wave speed. Means, depths and tangential velocities are unchanged;
// z = 1 for supercritical traces and thin or dry sides (model.mjs: lowFroudeVelocities).
fn lowFroudeVelocities(hl: f32, hr: f32, ul: f32, ur: f32, vl: f32, vr: f32) -> vec2<f32> {
    if p.lowFroude == 0u || hl <= H_THIN || hr <= H_THIN { return vec2<f32>(ul, ur); }
    let fr = max(sqrt(ul * ul + vl * vl) / sqrt(p.gravity * hl), sqrt(ur * ur + vr * vr) / sqrt(p.gravity * hr));
    let z = min(1.0, fr);
    let m = 0.5 * (ul + ur); let d = 0.5 * (ul - ur);
    return vec2<f32>(m + z * d, m - z * d);
}

fn faceFlux(lIn: vec4<f32>, rIn: vec4<f32>, axis: u32, codeIn: i32, kl: u32, kr: u32, slot: u32, t0: f32, t1: f32) -> FaceFlux {
    if codeIn == 3 { return FaceFlux(vec3<f32>(0.0), 0.0, 0.0, false); }
    var l = lIn; var r = rIn; var zl = bedTile[kl]; var zr = bedTile[kr]; var hcl = depthTile[kl]; var hcr = depthTile[kr];
    var code = codeIn;
    let hv = hydrographMean(slot, t0, t1);
    var mass = 0.0; var prescribed = false;
    if hv.x != 0.0 {
        let low = code < 0;
        let ghost = ghostTrace(hv, select(l, r, low), select(hcl, hcr, low), select(zl, zr, low), axis, select(-1.0, 1.0, low));
        if low { l = ghost.trace; zl = zr; hcl = hcr; } else { r = ghost.trace; zr = zl; hcr = hcl; }
        mass = ghost.mass; prescribed = ghost.prescribed;
    } else {
        if code == -4 { code = select(-2, -1, r[axis + 1u] > 0.0); } // outflow: reflect if inward
        if code == 4 { code = select(2, 1, l[axis + 1u] < 0.0); }
        if code < 0 { l = r; zl = zr; hcl = hcr; if code == -1 { l[axis + 1u] = -l[axis + 1u]; } }
        if code > 0 { r = l; zr = zl; hcr = hcl; if code == 1 { r[axis + 1u] = -r[axis + 1u]; } }
        prescribed = code == 1 || code == -1; // reflective faces are exactly impermeable
    }
    let cut = hydrostaticDepths(hcl, zl, l, hcr, zr, r); let hl = cut.x; let hr = cut.y;
    var un = vec2<f32>(l[axis + 1u], r[axis + 1u]);
    if codeIn == 0 && hv.x == 0.0 { un = lowFroudeVelocities(hl, hr, un.x, un.y, l[2u - axis], r[2u - axis]); }
    let normal = hllc(vec3<f32>(hl, un.x, l[2u - axis]), vec3<f32>(hr, un.y, r[2u - axis]));
    var f = normal.f;
    if axis == 1u { f = normal.f.xzy; }
    if prescribed { f.x = mass; }
    return FaceFlux(f, 0.5 * p.gravity * hl * hl, 0.5 * p.gravity * hr * hr, normal.fallback);
}

// Tracer (model.mjs: cellConcentration; muscl-reference.mjs: prepareTracer; reference.mjs: tracerFlux).
fn concentration(h: f32, hc: f32) -> f32 {
    if h > 0.0 { return clamp(hc / h, p.cmin, p.cmax); }
    return 0.0;
}
// Concentration traces (W, E, S, N) of tile cell k: MC slopes with the Hancock predictor of
// c_t + u c_x + v c_y = 0, clipped to the range of the cell and its neighbours; first order where
// the flow reconstruction is. Reads cell concentrations (phase 1 contents of cTile).
fn tracerTraces(k: u32, halfDt: f32, first: bool) -> vec4<f32> {
    let c = cTile[k];
    if solidTile[k] != 0u || first { return vec4<f32>(c); }
    let cw = select(cTile[k - 1u], c, solidTile[k - 1u] != 0u);
    let ce = select(cTile[k + 1u], c, solidTile[k + 1u] != 0u);
    let cs = select(cTile[k - PX], c, solidTile[k - PX] != 0u);
    let cn = select(cTile[k + PX], c, solidTile[k + PX] != 0u);
    let sx = mc(cw, c, ce); let sy = mc(cs, c, cn);
    let d = -halfDt * (tile[k].y * sx / p.dx + tile[k].z * sy / p.dy);
    let lo = min(min(c, min(cw, ce)), min(cs, cn)); let hi = max(max(c, max(cw, ce)), max(cs, cn));
    return clamp(vec4<f32>(c - 0.5 * sx + d, c + 0.5 * sx + d, c - 0.5 * sy + d, c + 0.5 * sy + d), vec4<f32>(lo), vec4<f32>(hi));
}
// Diffusive tracer flux -D min(hL, hR) (cR - cL) / d between tile cells kl and kr (cell values), with
// D the molecular diffusivity plus the turbulent part nu_f / Sc_t.
fn diffusive(kl: u32, kr: u32, d: f32, nuFace: f32) -> f32 {
    return -(p.diffusivity + nuFace * p.invSchmidt) * min(depthTile[kl], depthTile[kr]) * (cTile[kr] - cTile[kl]) / d;
}
// Eddy viscosity of tile cell k from the phase 1 tile (velocities mapped and reflected as for the
// reconstruction); (gx0, gy0) is the global cell of tile index 0 (reference.mjs: computeEddyViscosity).
fn eddy(k: u32, gx0: i32, gy0: i32) -> f32 {
    let h = tile[k].x;
    if solidTile[k] != 0u || !(h > H_THIN) { return 0.0; }
    let n = cells[u32(mapY(gy0 + i32(k / PX))) * p.width + u32(mapX(gx0 + i32(k % PX)))].y;
    var nu = localNu(h, length(tile[k].yz), n);
    if p.smagorinsky > 0.0 {
        let ux = (tile[k + 1u].y - tile[k - 1u].y) / (2.0 * p.dx); let vx = (tile[k + 1u].z - tile[k - 1u].z) / (2.0 * p.dx);
        let uy = (tile[k + PX].y - tile[k - PX].y) / (2.0 * p.dy); let vy = (tile[k + PX].z - tile[k - PX].z) / (2.0 * p.dy);
        let m = p.smagorinsky * min(p.dx, p.dy);
        nu += m * m * sqrt(2.0 * ux * ux + 2.0 * vy * vy + (uy + vx) * (uy + vx));
    }
    return nu;
}
// Viscous momentum flux -h_f nu_f (u_R - u_L) / d between tile cells (global x, y components).
fn viscousFlux(kl: u32, kr: u32, d: f32, nuFace: f32) -> vec2<f32> {
    let hl = depthTile[kl]; let hr = depthTile[kr];
    if !(hl > H_THIN) || !(hr > H_THIN) { return vec2<f32>(0.0); }
    return -(min(hl, hr) * nuFace / d) * (tile[kr].yz - tile[kl].yz);
}
// Upwind tracer flux with the face mass flux; a ghost side copies the interior concentration, or
// carries the hydrograph segment's concentration on an inflow face.
fn tracerFlux(clIn: f32, crIn: f32, code: i32, slot: u32, mass: f32) -> f32 {
    if code == 3 { return 0.0; }
    var cl = clIn; var cr = crIn;
    let s = seriesOf(slot);
    if code < 0 { cl = select(cr, s.conc, s.kind != 0.0); }
    if code > 0 { cr = select(cl, s.conc, s.kind != 0.0); }
    return mass * select(cr, cl, mass > 0.0);
}

fn accumulate(slot: u32, value: f32) {
    let a = boundary[slot]; let y = value - a.y; let t = a.x + y;
    boundary[slot] = vec2<f32>(t, (t - a.x) - y);
}

fn tileIndex(lx: u32, ly: u32) -> u32 { return (ly + 2u) * PX + lx + 2u; }

@compute @workgroup_size(16, 16)
fn fusedStep(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) li: u32) {
    if li == 0u {
        wgTime = vec4<f32>(bitcast<f32>(atomicLoad(&control[0])), bitcast<f32>(atomicLoad(&control[11])),
            bitcast<f32>(atomicLoad(&control[12])), 0.0);
        atomicStore(&wgMax[0], 0u); atomicStore(&wgMax[1], 0u);
        atomicStore(&wgCount[0], 0u); atomicStore(&wgCount[1], 0u); atomicStore(&wgCount[2], 0u);
    }
    let times = workgroupUniformLoad(&wgTime);
    let dt = times.x;
    let start = times.yz;
    let tStart = clockTime(start, 0.0);                     // faces use the hydrograph mean over
    let tEnd = clockTime(clockAdvance(start, dt), 0.0);    // [tStart, tEnd]; tEnd also bounds the next step
    let w = p.width; let h = p.height;
    let ox = wid.x * TX; let oy = wid.y * TY;
    let lx = lid.x; let ly = lid.y;
    let gx = ox + lx; let gy = oy + ly;
    let inside = gx < w && gy < h;
    let i = gy * w + gx;
    if dt == 0.0 {
        // The batch reached its target time; carry the state through the ping-pong pair.
        if inside {
            stateOut[i] = stateIn[i];
            if p.tracer != 0u { tracerOut[i] = tracerIn[i]; }
        }
        return;
    }

    // Phase 1: load the tile with the multi-pass boundary mapping and wall reflection.
    for (var k = li; k < PX * PY; k += TX * TY) {
        let x = i32(ox) + i32(k % PX) - 2; let y = i32(oy) + i32(k / PX) - 2;
        let c = u32(mapY(y)) * w + u32(mapX(x));
        let u = stateIn[c]; let cell = cells[c];
        var v = velocity(u);
        if p.boundaryX == 0u && (x < 0 || x >= i32(w)) { v.x = -v.x; }
        if p.boundaryY == 0u && (y < 0 || y >= i32(h)) { v.y = -v.y; }
        tile[k] = vec4<f32>(u.x, v, cell.x);
        solidTile[k] = select(0u, 1u, cell.w > 0.5);
        bedTile[k] = cell.x;
        depthTile[k] = u.x;
        if p.tracer != 0u { cTile[k] = concentration(u.x, tracerIn[c].x); }
    }
    workgroupBarrier();

    // Phase 2: own traces in registers; 64 threads also reconstruct one halo cell.
    let halfDt = 0.5 * dt;
    let own = reconstruct(lx + 2u, ly + 2u, halfDt);
    let tracing = p.tracer != 0u;
    var ownC = vec4<f32>(0.0);
    if tracing { ownC = tracerTraces(tileIndex(lx, ly), halfDt, own.fallback); }
    var halo = vec4<f32>(0.0); var haloC = 0.0;
    if li < 64u {
        let side = li / 16u; let j = li % 16u;
        if side == 0u {
            let t = reconstruct(1u, j + 2u, halfDt); halo = t.e;
            if tracing { haloC = tracerTraces((j + 2u) * PX + 1u, halfDt, t.fallback).y; }
        } else if side == 1u {
            let t = reconstruct(TX + 2u, j + 2u, halfDt); halo = t.w;
            if tracing { haloC = tracerTraces((j + 2u) * PX + TX + 2u, halfDt, t.fallback).x; }
        } else if side == 2u {
            let t = reconstruct(j + 2u, 1u, halfDt); halo = t.n;
            if tracing { haloC = tracerTraces(PX + j + 2u, halfDt, t.fallback).w; }
        } else {
            let t = reconstruct(j + 2u, TY + 2u, halfDt); halo = t.s;
            if tracing { haloC = tracerTraces((TY + 2u) * PX + j + 2u, halfDt, t.fallback).z; }
        }
    }
    // Eddy viscosity, viscous momentum fluxes and diffusive tracer fluxes from the phase 1 tile and
    // cell concentrations, before the tiles hold traces. Edge threads also need the east/north faces.
    var dW = 0.0; var dS = 0.0; var dE = 0.0; var dN = 0.0;
    var vW = vec2<f32>(0.0); var vS = vW; var vE = vW; var vN = vW;
    var nuW = 0.0; var nuS = 0.0; var nuE = 0.0; var nuN = 0.0;
    let gx0 = i32(ox) - 2; let gy0 = i32(oy) - 2;
    if viscous() {
        let k = tileIndex(lx, ly); let nuMe = eddy(k, gx0, gy0);
        nuW = 0.5 * (eddy(k - 1u, gx0, gy0) + nuMe); nuS = 0.5 * (eddy(k - PX, gx0, gy0) + nuMe);
        vW = viscousFlux(k - 1u, k, p.dx, nuW); vS = viscousFlux(k - PX, k, p.dy, nuS);
        if lx == TX - 1u { nuE = 0.5 * (nuMe + eddy(k + 1u, gx0, gy0)); vE = viscousFlux(k, k + 1u, p.dx, nuE); }
        if ly == TY - 1u { nuN = 0.5 * (nuMe + eddy(k + PX, gx0, gy0)); vN = viscousFlux(k, k + PX, p.dy, nuN); }
    }
    if tracing && (p.diffusivity > 0.0 || viscous()) {
        let k = tileIndex(lx, ly);
        dW = diffusive(k - 1u, k, p.dx, nuW); dS = diffusive(k - PX, k, p.dy, nuS);
        dE = diffusive(k, k + 1u, p.dx, nuE); dN = diffusive(k, k + PX, p.dy, nuN);
    }
    workgroupBarrier();
    tile[li] = own.e;
    tile[256u + li] = own.n;
    if li < 64u { tile[512u + li] = halo; }
    if tracing {
        cTile[li] = ownC.y;
        cTile[256u + li] = ownC.w;
        if li < 64u { cTile[512u + li] = haloC; }
    }
    workgroupBarrier();

    // Phase 3: west/south faces of each thread; extra east/north faces on the tile edge.
    let me = tileIndex(lx, ly);
    let solidMe = solidTile[me] != 0u;
    var hlleCount = 0u;
    var fw = FaceFlux(vec3<f32>(0.0), 0.0, 0.0, false);
    var fs = fw; var fe = fw; var fN = fw;
    var gw = 0.0; var gs = 0.0; var ge = 0.0; var gn = 0.0; // tracer face fluxes
    if gx <= w && gy < h {
        var left = tile[512u + ly];
        if lx > 0u { left = tile[li - 1u]; }
        let code = faceCode(p.boundaryX, gx == 0u, gx == w, solidTile[me - 1u] != 0u, solidMe);
        fw = faceFlux(left, own.w, 0u, code, me - 1u, me, slotX(gx, gy), tStart, tEnd);
        if code == 0 { fw.flux = vec3<f32>(fw.flux.x, fw.flux.yz + vW); }
        hlleCount += select(0u, 1u, fw.fallback);
        if tracing {
            var cl = cTile[512u + ly];
            if lx > 0u { cl = cTile[li - 1u]; }
            gw = tracerFlux(cl, ownC.x, code, slotX(gx, gy), fw.flux.x) + select(0.0, dW, code == 0);
        }
    }
    if gy <= h && gx < w {
        var left = tile[544u + lx];
        if ly > 0u { left = tile[256u + li - TX]; }
        let code = faceCode(p.boundaryY, gy == 0u, gy == h, solidTile[me - PX] != 0u, solidMe);
        fs = faceFlux(left, own.s, 1u, code, me - PX, me, slotY(gx, gy), tStart, tEnd);
        if code == 0 { fs.flux = vec3<f32>(fs.flux.x, fs.flux.yz + vS); }
        hlleCount += select(0u, 1u, fs.fallback);
        if tracing {
            var cl = cTile[544u + lx];
            if ly > 0u { cl = cTile[256u + li - TX]; }
            gs = tracerFlux(cl, ownC.z, code, slotY(gx, gy), fs.flux.x) + select(0.0, dS, code == 0);
        }
    }
    if lx == TX - 1u && inside {
        let code = faceCode(p.boundaryX, false, gx + 1u == w, solidMe, solidTile[me + 1u] != 0u);
        fe = faceFlux(own.e, tile[528u + ly], 0u, code, me, me + 1u, slotX(gx + 1u, gy), tStart, tEnd);
        if code == 0 { fe.flux = vec3<f32>(fe.flux.x, fe.flux.yz + vE); }
        hlleCount += select(0u, 1u, fe.fallback && gx + 1u == w);
        if tracing { ge = tracerFlux(ownC.y, cTile[528u + ly], code, slotX(gx + 1u, gy), fe.flux.x) + select(0.0, dE, code == 0); }
    }
    if ly == TY - 1u && inside {
        let code = faceCode(p.boundaryY, false, gy + 1u == h, solidMe, solidTile[me + PX] != 0u);
        fN = faceFlux(own.n, tile[560u + lx], 1u, code, me, me + PX, slotY(gx, gy + 1u), tStart, tEnd);
        if code == 0 { fN.flux = vec3<f32>(fN.flux.x, fN.flux.yz + vN); }
        hlleCount += select(0u, 1u, fN.fallback && gy + 1u == h);
        if tracing { gn = tracerFlux(ownC.w, cTile[560u + lx], code, slotY(gx, gy + 1u), fN.flux.x) + select(0.0, dN, code == 0); }
    }
    workgroupBarrier();
    tile[li] = vec4<f32>(fw.flux, fw.pl);
    tile[256u + li] = vec4<f32>(fs.flux, fs.pl);
    if tracing { cTile[li] = gw; cTile[256u + li] = gs; }
    workgroupBarrier();

    // Phase 4: conservative update, sources, friction, next-step wave speeds.
    var errors = 0u;
    if inside {
        var result = vec3<f32>(0.0); var comp = 0.0; var amount = vec2<f32>(0.0);
        if !solidMe {
            var east = vec4<f32>(fe.flux, fe.pl);
            if lx < TX - 1u { east = tile[li + 1u]; }
            var north = vec4<f32>(fN.flux, fN.pl);
            if ly < TY - 1u { north = tile[256u + li + TX]; }
            let cell = cells[i];
            var df = east.xyz - fw.flux; var dg = north.xyz - fs.flux;
            let hc = depthTile[me];
            let hx = 0.5 * ((hc + (own.w.x - own.w.w)) + (hc + (own.e.x - own.e.w)));
            let hy = 0.5 * ((hc + (own.s.x - own.s.w)) + (hc + (own.n.x - own.n.w)));
            df.y = (east.y - east.w) - (fw.flux.y - fw.pr) + p.gravity * hx * (own.e.x - own.w.x);
            dg.z = (north.z - north.w) - (fs.flux.z - fs.pr) + p.gravity * hy * (own.n.x - own.s.x);
            let old = stateIn[i];
            result = vec3<f32>(0.0, old.yz - dt / p.dx * df.yz - dt / p.dy * dg.yz);
            // Compensated (Kahan) depth update; old.w carries the low-order bits of depth.
            let dh = -(dt / p.dx) * df.x - (dt / p.dy) * dg.x + (p.rain + cell.z) * dt;
            let dhc = dh - old.w; result.x = old.x + dhc; comp = (result.x - old.x) - dhc;
            if result.x == 0.0 { result.y = 0.0; result.z = 0.0; }
            // Wind stress on wet cells, before friction (model.mjs: windStress).
            if result.x > H_THIN { result.y += dt * p.windX; result.z += dt * p.windY; }
            if p.coriolis != 0.0 && result.x > H_THIN {
                // Coriolis: exact rotation of q by -f dt (model.mjs: coriolisRotate).
                let cs = rotation(p.coriolis * dt); let c = cs.x; let s = cs.y;
                result = vec3<f32>(result.x, c * result.y + s * result.z, -s * result.y + c * result.z);
            }
            if cell.y > 0.0 && result.x > 0.0 {
                let r = pow(result.x, 7.0 / 3.0); let d = r + dt * p.gravity * cell.y * cell.y * length(result.yz);
                if d > 0.0 { result.y *= r / d; result.z *= r / d; }
            }
            if result.x < H_THIN {
                // q <- q * sqrt(2) h^2 / sqrt(h^4 + H^4): continuous, equal to 1 at h = H.
                let h2 = result.x * result.x;
                result.y *= 1.41421356 * h2 / sqrt(h2 * h2 + H_THIN4);
                result.z *= 1.41421356 * h2 / sqrt(h2 * h2 + H_THIN4);
            }
            if result.x < 0.0 || !finite(result.x) || !finite(result.y) || !finite(result.z) { errors = 4u; }
            if p.boundaryX != 1u {
                if gx == 0u { accumulate(gy, -fw.flux.x * dt * p.dy); }
                if gx == w - 1u { accumulate(h + gy, east.x * dt * p.dy); }
            }
            if p.boundaryY != 1u {
                if gy == 0u { accumulate(2u * h + gx, -fs.flux.x * dt * p.dx); }
                if gy == h - 1u { accumulate(2u * h + w + gx, north.x * dt * p.dx); }
            }
            if tracing {
                var te = ge; if lx < TX - 1u { te = cTile[li + 1u]; }
                var tn = gn; if ly < TY - 1u { tn = cTile[256u + li + TX]; }
                // Compensated update of h c; a cell that dries completely keeps no tracer.
                let old = tracerIn[i];
                let dc = (-(dt / p.dx) * (te - gw) - (dt / p.dy) * (tn - gs) + cell.z * tracerSource[i] * dt) - old.y;
                let t = old.x + dc;
                amount = vec2<f32>(t, (t - old.x) - dc);
                if result.x == 0.0 { amount = vec2<f32>(0.0); }
                if p.settleVelocity > 0.0 && result.x > 0.0 && amount.x > 0.0 {
                    // Exact local settling at the new depth (model.mjs: settledConcentration).
                    let c = amount.x / result.x; var factor = exp(-p.settleVelocity * dt / result.x);
                    if p.settleExponent > 0.0 {
                        factor = pow(1.0 + p.settleExponent * p.settleVelocity * dt * pow(c, p.settleExponent) / result.x, -1.0 / p.settleExponent);
                    }
                    amount = vec2<f32>(amount.x * factor, 0.0);
                }
                if !finite(amount.x) { errors = 4u; }
                let b = 2u * (w + h);
                if p.boundaryX != 1u {
                    if gx == 0u { accumulate(b + gy, -gw * dt * p.dy); }
                    if gx == w - 1u { accumulate(b + h + gy, te * dt * p.dy); }
                }
                if p.boundaryY != 1u {
                    if gy == 0u { accumulate(b + 2u * h + gx, -gs * dt * p.dx); }
                    if gy == h - 1u { accumulate(b + 2u * h + w + gx, tn * dt * p.dx); }
                }
            }
            if own.fallback { atomicAdd(&wgCount[1], 1u); }
        }
        if p.faultDt > 0.0 && i == 0u && dt > p.faultDt && tStart >= p.faultTime { errors |= 4u; }
        let next = vec4<f32>(result, comp);
        stateOut[i] = next;
        if tracing { tracerOut[i] = amount; }
        // Only valid states contribute next-step speeds; an invalid update reports bit 4 alone,
        // which the host treats as retryable at a smaller dt.
        if errors == 0u {
            let speed = max(speeds(next) + vec2<f32>(viscousSpeed(next, cells[i].y)), boundarySpeeds(gx, gy, next, bedTile[me], tEnd));
            if finite(speed.x) && finite(speed.y) {
                atomicMax(&wgMax[0], bitcast<u32>(speed.x));
                atomicMax(&wgMax[1], bitcast<u32>(speed.y));
            } else { errors |= 1u; }
        }
        if hlleCount > 0u { atomicAdd(&wgCount[0], hlleCount); }
        if errors != 0u { atomicOr(&wgCount[2], errors); }
    } else if hlleCount > 0u { atomicAdd(&wgCount[0], hlleCount); }
    workgroupBarrier();
    if li == 0u {
        let slot = 3u - 2u * p.parity;
        atomicMax(&control[slot], atomicLoad(&wgMax[0]));
        atomicMax(&control[slot + 1u], atomicLoad(&wgMax[1]));
        atomicAdd(&control[6], atomicLoad(&wgCount[0]));
        atomicAdd(&control[7], atomicLoad(&wgCount[1]));
        let errors = atomicLoad(&wgCount[2]);
        // The host replays the valid prefix of a failed batch and retries only this step.
        if errors != 0u { atomicOr(&control[5], errors); atomicMin(&control[10], atomicLoad(&control[8]) - 1u); }
    }
}
