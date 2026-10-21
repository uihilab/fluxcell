// Round piles and islands in 3D: smooth cylinders (48 sides and a top cap) at their true radius, in
// place of the raised solid cells, which would show the cell staircase.
struct Scene { viewProj: mat4x4<f32>, eye: vec4<f32>, params: vec4<f32>, mesh: vec4<u32> }
@group(0) @binding(0) var<uniform> scene: Scene;
// Per cylinder: centre x, y (m from the domain centre), radius (m), and bottom and top (scene z).
@group(0) @binding(1) var<storage, read> cylinders: array<array<f32, 5>>;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) world: vec3<f32> }
const SIDES = 48u;
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> Out {
    let c = cylinders[k];
    var o: Out;
    var corner: vec3<f32>;
    if vi < SIDES * 6u {
        // Side quad of segment s: two triangles.
        let s = vi / 6u; let j = vi % 6u;
        var pick = array<vec2<u32>, 6>(vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 1u));
        let a = 6.2831853 * f32(s + pick[j].x) / f32(SIDES);
        corner = vec3<f32>(cos(a), sin(a), f32(pick[j].y));
        o.normal = vec3<f32>(cos(a), sin(a), 0.0);
    } else {
        // Top cap fan.
        let t = vi - SIDES * 6u; let s = t / 3u; let j = t % 3u;
        let a = 6.2831853 * f32(s + select(0u, 1u, j == 2u)) / f32(SIDES);
        corner = select(vec3<f32>(cos(a), sin(a), 1.0), vec3<f32>(0.0, 0.0, 1.0), j == 0u);
        o.normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    o.world = vec3<f32>(c[0] + c[2] * corner.x, c[1] + c[2] * corner.y, mix(c[3], c[4], corner.z));
    o.pos = scene.viewProj * vec4<f32>(o.world, 1.0);
    return o;
}
@fragment fn fs(o: Out) -> @location(0) vec4<f32> {
    let n = normalize(o.normal); let l = normalize(vec3<f32>(-0.45, 0.55, 0.70)); let v = normalize(scene.eye.xyz - o.world);
    let base = vec3<f32>(0.62, 0.62, 0.60);
    let lit = base * (0.35 + 0.65 * max(dot(n, l), 0.0)) + vec3<f32>(0.12) * pow(max(dot(reflect(-l, n), v), 0.0), 24.0);
    return vec4<f32>(lit, 1.0);
}
