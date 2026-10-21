// Bed evolution for the scour demo (viewer-side, between solver batches): Meyer-Peter and Mueller
// (1948) bedload q* = 8 (theta - theta_c)^1.5, theta = g n² |u|² / (h^(1/3) (s - 1) g d), along the
// velocity; face fluxes upwinded, with a downslope term -beta |q| dz/dx; Exner update
// dz = -dt MF / (1 - p) div q, conservative by construction (face fluxes; none through solid faces
// or the edges, whose cells keep their own flux); then a slide that moves sediment across faces
// steeper than the angle of repose. The water depth is kept (water volume conserved).
//#include view.wgsl
struct Morph { dtm: f32, d: f32, s: f32, porosity: f32, thetaC: f32, beta: f32, repose: f32, pad: f32 }
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> faces: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> m: Morph;
fn idx(x: i32, y: i32) -> u32 { return u32(y) * view.width + u32(x); }
fn open(x: i32, y: i32) -> bool { return x >= 0 && y >= 0 && x < i32(view.width) && y < i32(view.height) && cells[idx(x, y)].w < 0.5; }
fn bedload(i: u32) -> vec2<f32> {
    let st = state[i]; let h = st.x;
    if h <= view.wetDepth || cells[i].w > 0.5 { return vec2<f32>(0.0); }
    let u = st.yz / h; let speed = length(u); let n = cells[i].y;
    let theta = n * n * speed * speed / (pow(h, 1.0 / 3.0) * (m.s - 1.0) * m.d);
    if theta <= m.thetaC || speed <= 0.0 { return vec2<f32>(0.0); }
    let q = 8.0 * pow(theta - m.thetaC, 1.5) * sqrt((m.s - 1.0) * 9.81 * m.d * m.d * m.d);
    return q * u / speed;
}
// Flux through the east (x) or north (y) face of cell (x, y); zero next to solid cells or past an edge.
fn faceFlux(x: i32, y: i32, axis: i32) -> f32 {
    let ox = select(0, 1, axis == 0); let oy = select(0, 1, axis == 1);
    if !open(x, y) || !open(x + ox, y + oy) { return 0.0; }
    let a = bedload(idx(x, y)); let b = bedload(idx(x + ox, y + oy));
    let qa = select(a.y, a.x, axis == 0); let qb = select(b.y, b.x, axis == 0);
    let up = select(qb, qa, qa + qb > 0.0);
    let spacing = select(view.dy, view.dx, axis == 0);
    let slope = (cells[idx(x + ox, y + oy)].x - cells[idx(x, y)].x) / spacing;
    return up - m.beta * 0.5 * (length(a) + length(b)) * slope;
}
@compute @workgroup_size(16, 16) fn fluxes(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height { return; }
    faces[id.y * view.width + id.x] = vec2<f32>(faceFlux(i32(id.x), i32(id.y), 0), faceFlux(i32(id.x), i32(id.y), 1));
}
// Avalanche: across faces steeper than the repose slope, half the excess height moves downhill.
fn slideFlux(x: i32, y: i32, axis: i32) -> f32 {
    let ox = select(0, 1, axis == 0); let oy = select(0, 1, axis == 1);
    if !open(x, y) || !open(x + ox, y + oy) { return 0.0; }
    let spacing = select(view.dy, view.dx, axis == 0);
    let dz = cells[idx(x + ox, y + oy)].x - cells[idx(x, y)].x; let excess = abs(dz) - m.repose * spacing;
    if excess <= 0.0 { return 0.0; }
    return -sign(dz) * 0.25 * excess * spacing; // volume per unit width moved from the higher to the lower cell
}
@compute @workgroup_size(16, 16) fn slides(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height { return; }
    faces[id.y * view.width + id.x] = vec2<f32>(slideFlux(i32(id.x), i32(id.y), 0), slideFlux(i32(id.x), i32(id.y), 1));
}
fn divergence(x: i32, y: i32) -> f32 {
    let i = idx(x, y); var west = 0.0; var south = 0.0;
    if x > 0 { west = faces[idx(x - 1, y)].x; }
    if y > 0 { south = faces[idx(x, y - 1)].y; }
    return (faces[i].x - west) / view.dx + (faces[i].y - south) / view.dy;
}
@compute @workgroup_size(16, 16) fn exner(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height { return; }
    let i = idx(i32(id.x), i32(id.y));
    if cells[i].w > 0.5 { return; }
    cells[i].x -= m.dtm / (1.0 - m.porosity) * divergence(i32(id.x), i32(id.y));
}
@compute @workgroup_size(16, 16) fn settle(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height { return; }
    let i = idx(i32(id.x), i32(id.y));
    if cells[i].w > 0.5 { return; }
    cells[i].x -= divergence(i32(id.x), i32(id.y)) * view.dx * view.dy / (view.dx * view.dy) ;
}
