// Waterline field, computed per cell before each frame: the signed field of nodeGap, scaled by four
// wet depths and clamped to [-1, 1], then smoothed over 3 x 3 cells ([1 2 1] x [1 2 1] / 16). The
// zero contour of the smoothed field, interpolated between centres, is the drawn waterline: a
// smooth curve through a front that the solver resolves to one cell, not a staircase of cells.
//#include view.wgsl
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> raw: array<f32>;
@group(0) @binding(4) var<storage, read_write> smoothed: array<f32>;
@group(0) @binding(5) var<storage, read_write> levels: array<f32>;
@group(0) @binding(6) var<storage, read_write> slopes: array<vec2<f32>>;
// Water level of a cell: its surface if wet; the mean level of its wet neighbours if it is dry (or
// solid) next to water; its bed otherwise (never drawn as water).
fn levelAt(x: i32, y: i32) -> f32 {
    let i = cellAt(x, y); let wetMin = 0.25 * view.wetDepth;
    if cells[i].w < 0.5 && state[i].x > wetMin { return cells[i].x + state[i].x; }
    var n = 0.0; var sum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let j = cellAt(x + dx, y + dy);
            if cells[j].w < 0.5 && state[j].x > wetMin { n += 1.0; sum += cells[j].x + state[j].x; }
        }
    }
    if n == 0.0 { return cells[i].x; }
    return sum / n;
}
//#include grid.wgsl
// Waterline field at a cell centre, positive in water and negative on land; the waterline is its
// zero contour, interpolated between centres. Wet cells give their depth. A dry cell next to water
// gives the level of its wet neighbours minus its own bed (negative, since it stayed dry), so the
// line falls where the water surface meets the ground between the two centres. Every value belongs
// to one cell, so the field is continuous and the line has no steps at cell edges. Solid cells take
// the depth of their wet neighbours (the outline is drawn over them).
fn nodeGap(x: i32, y: i32) -> f32 {
    let i = cellAt(x, y); let solid = cells[i].w > 0.5; let wetMin = 0.25 * view.wetDepth;
    if !solid && state[i].x > wetMin { return state[i].x; }
    var n = 0.0; var sum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let j = cellAt(x + dx, y + dy);
            if cells[j].w < 0.5 && state[j].x > wetMin { n += 1.0; sum += select(cells[j].x + state[j].x, state[j].x, solid); }
        }
    }
    if n == 0.0 { return -1.0; }
    if solid { return sum / n; }
    return min(sum / n - cells[i].x, -0.1 * view.wetDepth);
}
@compute @workgroup_size(16, 16) fn signed(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height { return; }
    raw[id.y * view.width + id.x] = clamp(nodeGap(i32(id.x), i32(id.y)) / (4.0 * view.wetDepth), -1.0, 1.0);
    levels[id.y * view.width + id.x] = levelAt(i32(id.x), i32(id.y));
}
@compute @workgroup_size(16, 16) fn blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.width || id.y >= view.height { return; }
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            sum += f32((2 - abs(dx)) * (2 - abs(dy))) * raw[cellAt(i32(id.x) + dx, i32(id.y) + dy)];
        }
    }
    smoothed[id.y * view.width + id.x] = sum / 16.0;
    let x = i32(id.x); let y = i32(id.y);
    slopes[id.y * view.width + id.x] = vec2<f32>((levels[cellAt(x + 1, y)] - levels[cellAt(x - 1, y)]) / (2.0 * view.dx),
                                                 (levels[cellAt(x, y + 1)] - levels[cellAt(x, y - 1)]) / (2.0 * view.dy));
}
