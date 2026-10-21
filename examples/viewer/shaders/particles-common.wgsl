// Flow tracers (visual): particles carried by the depth-averaged velocity. Each is (x, y) in cell
// coordinates (centres at integers), its age and lifetime in frames.
//#include view.wgsl
struct Scene { viewProj: mat4x4<f32>, eye: vec4<f32>, params: vec4<f32>, mesh: vec4<u32> }
struct Particles { dt: f32, frame: u32, count: u32, width: f32, colour: vec4<f32>, seed: vec4<f32>, life: vec2<f32>, pad: vec2<f32> }
// seed: the region (fractions x0, y0, x1, y1 of the domain) where particles are released; life:
// minimum and range of lifetimes (frames).
//#include grid.wgsl
fn velAt(i: u32) -> vec2<f32> {
    let s = state[i];
    if s.x > view.wetDepth && cells[i].w < 0.5 { return s.yz / s.x; }
    return vec2<f32>(0.0);
}
// Bilinear velocity (m/s) at a position in cell coordinates.
fn velocity(g: vec2<f32>) -> vec2<f32> {
    let b = floor(g); let f = g - b; let x0 = i32(b.x); let y0 = i32(b.y);
    return mix(mix(velAt(cellAt(x0, y0)), velAt(cellAt(x0 + 1, y0)), f.x), mix(velAt(cellAt(x0, y0 + 1)), velAt(cellAt(x0 + 1, y0 + 1)), f.x), f.y);
}
fn hash(v: u32) -> u32 { var x = v * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return (x >> 22u) ^ x; }
fn rand(v: u32) -> f32 { return f32(hash(v)) / 4294967295.0; }
