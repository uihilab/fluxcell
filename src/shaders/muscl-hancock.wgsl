// MUSCL-Hancock / HLLC (multi-pass reference path). Traces are [eta - eta_c, u, v, z - z_c]
// relative to each cell's own surface and bed: depth keeps full f32 precision at any elevation. Primitive traces [eta,u,v,z].
// Source-balanced prediction; shared face fluxes; immutable old state.
struct Params {
    width: u32, height: u32, boundaryX: u32, boundaryY: u32,
    dx: f32, dy: f32, gravity: f32, cfl: f32,
    maxDt: f32, rain: f32, clockLo: f32, clockHi: f32,
    // Words 12-17 are used by the fused kernel only; 18 is the low-Froude flag.
    pad0: f32, pad1: f32, pad2: f32, pad3: f32, pad4: f32, pad5: f32, lowFroude: u32, pad6: f32,
    // Words 20-23 are used by the fused kernel only; 24, 25: kinematic wind stress (m²/s²).
    pad7: f32, pad8: f32, pad9: f32, pad10: f32, windX: f32, windY: f32, pad11: f32, pad12: f32,
    // Word 28: Coriolis parameter f (1/s).
    coriolis: f32, pad13: f32, pad14: f32, pad15: f32,
}
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
struct Face { flux: vec4<f32>, pressure: vec4<f32> }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> stateIn: array<vec4<f32>>;
// [bed z (m), Manning n, source rate (m/s), solid (0/1)].
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> facesX: array<Face>;
@group(0) @binding(4) var<storage, read_write> facesY: array<Face>;
@group(0) @binding(5) var<storage, read_write> stateOut: array<vec4<f32>>;
// dt, max speed x/y, error bits, HLLE count, reconstruction fallback count.
@group(0) @binding(6) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> traces: array<vec4<f32>>;
// Compensated open-boundary outflow volume per boundary face slot (sum, compensation).
@group(0) @binding(8) var<storage, read_write> boundary: array<vec2<f32>>;

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn velocity(u: vec4<f32>) -> vec2<f32> {
    if u.x > 0.0 { return u.yz / u.x; }
    return vec2<f32>(0.0);
}
fn solid(i: u32) -> bool { return cells[i].w > 0.5; }

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
struct Series { kind: f32, scale: f32, start: u32, count: u32, still: f32 }
fn seriesOf(slot: u32) -> Series {
    if slot == NO_SLOT { return Series(0.0, 0.0, 0u, 0u, 0.0); }
    let b = 2u * (p.width + p.height);
    let segment = u32(boundary[2u * b + slot].x);
    if segment == 0u { return Series(0.0, 0.0, 0u, 0u, 0.0); }
    let head = boundary[3u * b + 3u * (segment - 1u)];
    let span = boundary[3u * b + 3u * (segment - 1u) + 1u];
    let extra = boundary[3u * b + 3u * (segment - 1u) + 2u];
    return Series(head.x, head.y, u32(span.x), u32(span.y), extra.y);
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

@compute @workgroup_size(64)
fn waveSpeeds(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.width * p.height { return; }
    let u = stateIn[id.x];
    let speed = max(abs(velocity(u)) + vec2<f32>(2.0 * sqrt(p.gravity * u.x)),
        boundarySpeeds(id.x % p.width, id.x / p.width, u, cells[id.x].x, clockTime(vec2<f32>(p.clockHi, p.clockLo), 0.0)));
    if !finite(speed.x) || !finite(speed.y) || u.x < 0.0 { atomicOr(&control[3], 1u); return; }
    atomicMax(&control[1], bitcast<u32>(speed.x));
    atomicMax(&control[2], bitcast<u32>(speed.y));
}

@compute @workgroup_size(1)
fn timestep() {
    let rate = bitcast<f32>(atomicLoad(&control[1])) / p.dx + bitcast<f32>(atomicLoad(&control[2])) / p.dy;
    var dt = p.maxDt;
    if rate > 0.0 { dt = min(dt, p.cfl / rate); }
    if !finite(rate) || !finite(dt) || dt <= 0.0 { atomicOr(&control[3], 2u); }
    atomicStore(&control[0], bitcast<u32>(dt));
}

fn cellIndex(x: i32, y: i32) -> u32 {
    let w = i32(p.width); let h = i32(p.height);
    var cx = clamp(x, 0, w - 1); var cy = clamp(y, 0, h - 1);
    if p.boundaryX == 1u { cx = (x + w) % w; }
    if p.boundaryY == 1u { cy = (y + h) % h; }
    return u32(cy * w + cx);
}

// [(h - hc) + (z - zc), u, v, z - zc] relative to the reconstructing cell (hc, zc).
fn primitive(x: i32, y: i32, hc: f32, zc: f32) -> vec4<f32> {
    let i = cellIndex(x, y); let u = stateIn[i]; let dz = cells[i].x - zc;
    var v = velocity(u);
    if p.boundaryX == 0u && (x < 0 || x >= i32(p.width)) { v.x = -v.x; }
    if p.boundaryY == 0u && (y < 0 || y >= i32(p.height)) { v.y = -v.y; }
    return vec4<f32>((u.x - hc) + dz, v, dz);
}

// Depth of a stencil neighbour; a solid neighbour mirrors the centre depth hc.
fn neighbourDepth(x: i32, y: i32, hc: f32) -> f32 {
    let i = cellIndex(x, y);
    if solid(i) { return hc; }
    return stateIn[i].x;
}

// A solid neighbour is the mirror image of the centre across the shared face.
fn neighbour(x: i32, y: i32, ox: i32, oy: i32, c: vec4<f32>, hc: f32, zc: f32) -> vec4<f32> {
    if !solid(cellIndex(x + ox, y + oy)) { return primitive(x + ox, y + oy, hc, zc); }
    var m = c;
    if ox != 0 { m.y = -m.y; } else { m.z = -m.z; }
    return m;
}

fn mc(a: f32, b: f32, c: f32) -> f32 {
    let dl = 2.0 * (b - a); let dc = 0.5 * (c - a); let dr = 2.0 * (c - b);
    if dl > 0.0 && dc > 0.0 && dr > 0.0 { return min(dl, min(dc, dr)); }
    if dl < 0.0 && dc < 0.0 && dr < 0.0 { return max(dl, max(dc, dr)); }
    return 0.0;
}

@compute @workgroup_size(64)
fn reconstruct(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.width * p.height { return; }
    let x = i32(i % p.width); let y = i32(i / p.width);
    let zc = cells[i].x; let h = stateIn[i].x; let c = primitive(x, y, h, zc);
    if solid(i) { for (var k = 0u; k < 4u; k++) { traces[4u * i + k] = c; } return; }
    let w = neighbour(x, y, -1, 0, c, h, zc); let e = neighbour(x, y, 1, 0, c, h, zc);
    let s = neighbour(x, y, 0, -1, c, h, zc); let n = neighbour(x, y, 0, 1, c, h, zc);
    let depths = min(min(neighbourDepth(x - 1, y, h), neighbourDepth(x + 1, y, h)),
        min(neighbourDepth(x, y - 1, h), neighbourDepth(x, y + 1, h)));
    var fallback = min(h, depths) <= H_THIN;
    // Bed steps larger than the depth: hydrostatic reconstruction and the Hancock predictor are
    // unreliable there (thin water on cliffs); reconstruct at first order.
    fallback = fallback || max(max(abs(w.w), abs(e.w)), max(abs(s.w), abs(n.w))) > h;
    var sx = vec4<f32>(0.0); var sy = vec4<f32>(0.0);
    if !fallback {
        for (var k = 0u; k < 4u; k++) { sx[k] = mc(w[k], c[k], e[k]); sy[k] = mc(s[k], c[k], n[k]); }
    }
    let slope = max(abs(sx.x - sx.w), abs(sy.x - sy.w));
    var theta = 1.0;
    if slope > 0.0 { theta = min(1.0, 2.0 * h / slope); }
    sx *= theta; sy *= theta;
    let halfDt = 0.5 * bitcast<f32>(atomicLoad(&control[0]));
    let delta = -halfDt * vec4<f32>(
        c.y * (sx.x - sx.w) / p.dx + c.z * (sy.x - sy.w) / p.dy + h * (sx.y / p.dx + sy.z / p.dy),
        c.y * sx.y / p.dx + c.z * sy.y / p.dy + p.gravity * sx.x / p.dx,
        c.y * sx.z / p.dx + c.z * sy.z / p.dy + p.gravity * sy.x / p.dy, 0.0);
    var faces = array<vec4<f32>, 4>(c - 0.5 * sx + delta, c + 0.5 * sx + delta,
        c - 0.5 * sy + delta, c + 0.5 * sy + delta);
    var invalid = false;
    for (var k = 0u; k < 4u; k++) {
        let f = faces[k];
        invalid = invalid || h + (f.x - f.w) < 0.0 || !finite(f.x) || !finite(f.y) || !finite(f.z) || !finite(f.w);
    }
    if invalid { for (var k = 0u; k < 4u; k++) { faces[k] = c; } fallback = true; }
    if fallback { atomicAdd(&control[5], 1u); }
    for (var k = 0u; k < 4u; k++) { traces[4u * i + k] = faces[k]; }
}

fn hlle(fl: vec3<f32>, fr: vec3<f32>, l: vec3<f32>, r: vec3<f32>, sl: f32, sr: f32) -> vec3<f32> {
    atomicAdd(&control[4], 1u);
    return (sr * fl - sl * fr + sl * sr * (r - l)) / (sr - sl);
}

fn hllc(lv: vec3<f32>, rv: vec3<f32>) -> vec3<f32> {
    let hl = lv.x; let hr = rv.x; let ul = lv.y; let ur = rv.y;
    if hl == 0.0 && hr == 0.0 { return vec3<f32>(0.0); }
    let cl = sqrt(p.gravity * hl); let cr = sqrt(p.gravity * hr);
    let pl = 0.5 * p.gravity * hl * hl; let pr = 0.5 * p.gravity * hr * hr;
    let l = vec3<f32>(hl, hl * lv.yz); let r = vec3<f32>(hr, hr * rv.yz);
    let fl = vec3<f32>(hl * ul, hl * ul * ul + pl, hl * ul * lv.z);
    let fr = vec3<f32>(hr * ur, hr * ur * ur + pr, hr * ur * rv.z);
    var sl = min(ul - cl, ur - cr); var sr = max(ul + cl, ur + cr);
    if hl == 0.0 { sl = ur - 2.0 * cr; sr = ur + cr; }
    if hr == 0.0 { sl = ul - cl; sr = ul + 2.0 * cl; }
    if sl >= 0.0 { return fl; }
    if sr <= 0.0 { return fr; }
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
    return fstar;
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
fn faceCode(boundaryType: u32, first: bool, last: bool, left: u32, right: u32) -> i32 {
    var code = 0;
    if (first || last) && boundaryType != 1u {
        var kind = 1;
        if boundaryType == 2u { kind = 2; }
        if boundaryType == 3u { kind = 4; }
        code = select(1, -1, first) * kind;
    }
    let sl = solid(left); let sr = solid(right);
    if sl && sr { return 3; }
    if sl { return -1; }
    if sr { return 1; }
    return code;
}

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

fn flux(left: u32, right: u32, axis: u32, codeIn: i32, slot: u32, t0: f32, t1: f32) -> Face {
    if codeIn == 3 { return Face(vec4<f32>(0.0), vec4<f32>(0.0)); }
    var l = traces[4u * left + 2u * axis + 1u]; var r = traces[4u * right + 2u * axis];
    var zl = cells[left].x; var zr = cells[right].x; var hcl = stateIn[left].x; var hcr = stateIn[right].x;
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
    var f = normal;
    if axis == 1u { f = normal.xzy; }
    if prescribed { f.x = mass; }
    return Face(vec4<f32>(f, 0.0), vec4<f32>(0.5 * p.gravity * hl * hl, 0.5 * p.gravity * hr * hr, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn faceFluxes(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; let w = p.width; let h = p.height;
    // Hydrograph interval of the step: from the clock to the clock advanced by dt.
    let clock = vec2<f32>(p.clockHi, p.clockLo);
    let t0 = clockTime(clock, 0.0); let t1 = clockTime(clockAdvance(clock, bitcast<f32>(atomicLoad(&control[0]))), 0.0);
    if i < (w + 1u) * h {
        let x = i % (w + 1u); let y = i / (w + 1u);
        var left = y * w + u32(max(i32(x) - 1, 0)); var right = y * w + min(x, w - 1u);
        if (x == 0u || x == w) && p.boundaryX == 1u { left = y * w + w - 1u; right = y * w; }
        facesX[i] = flux(left, right, 0u, faceCode(p.boundaryX, x == 0u, x == w, left, right), slotX(x, y), t0, t1);
    }
    if i < w * (h + 1u) {
        let x = i % w; let y = i / w;
        var left = u32(max(i32(y) - 1, 0)) * w + x; var right = min(y, h - 1u) * w + x;
        if (y == 0u || y == h) && p.boundaryY == 1u { left = (h - 1u) * w + x; right = x; }
        facesY[i] = flux(left, right, 1u, faceCode(p.boundaryY, y == 0u, y == h, left, right), slotY(x, y), t0, t1);
    }
}

fn bedCorrection(i: u32, axis: u32) -> f32 {
    let a = traces[4u * i + 2u * axis]; let b = traces[4u * i + 2u * axis + 1u]; let h = stateIn[i].x;
    let hbar = 0.5 * ((h + (a.x - a.w)) + (h + (b.x - b.w)));
    return p.gravity * hbar * (b.x - a.x);
}

fn accumulate(slot: u32, value: f32) {
    let a = boundary[slot]; let y = value - a.y; let t = a.x + y;
    boundary[slot] = vec2<f32>(t, (t - a.x) - y);
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; let w = p.width; let h = p.height;
    if i >= w * h { return; }
    if solid(i) { stateOut[i] = vec4<f32>(0.0); return; }
    let x = i % w; let y = i / w;
    let west = facesX[y * (w + 1u) + x]; let east = facesX[y * (w + 1u) + x + 1u];
    let south = facesY[y * w + x]; let north = facesY[(y + 1u) * w + x];
    let dt = bitcast<f32>(atomicLoad(&control[0]));
    let cell = cells[i];
    var df = east.flux.xyz - west.flux.xyz; var dg = north.flux.xyz - south.flux.xyz;
    df.y = (east.flux.y - east.pressure.x) - (west.flux.y - west.pressure.y) + bedCorrection(i, 0u);
    dg.z = (north.flux.z - north.pressure.x) - (south.flux.z - south.pressure.y) + bedCorrection(i, 1u);
    let old = stateIn[i];
    var result = vec3<f32>(0.0, old.yz - dt / p.dx * df.yz - dt / p.dy * dg.yz);
    // Compensated (Kahan) depth update; old.w carries the low-order bits of depth.
    let dh = -(dt / p.dx) * df.x - (dt / p.dy) * dg.x + (p.rain + cell.z) * dt;
    let dhc = dh - old.w; result.x = old.x + dhc; let comp = (result.x - old.x) - dhc;
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
    if result.x < 0.0 || !finite(result.x) || !finite(result.y) || !finite(result.z) { atomicOr(&control[3], 4u); }
    if p.boundaryX != 1u {
        if x == 0u { accumulate(y, -west.flux.x * dt * p.dy); }
        if x == w - 1u { accumulate(h + y, east.flux.x * dt * p.dy); }
    }
    if p.boundaryY != 1u {
        if y == 0u { accumulate(2u * h + x, -south.flux.x * dt * p.dx); }
        if y == h - 1u { accumulate(2u * h + w + x, north.flux.x * dt * p.dx); }
    }
    stateOut[i] = vec4<f32>(result, comp);
}
