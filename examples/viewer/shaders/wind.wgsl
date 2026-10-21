// Wind as a visual (not simulated): gust patches (cat's paws) drifting downwind over the water. pos in metres, wind in m/s, scale the domain size (m), t wall-clock s.
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i).x; let b = hash2(i + vec2<f32>(1.0, 0.0)).x;
    let c = hash2(i + vec2<f32>(0.0, 1.0)).x; let d = hash2(i + vec2<f32>(1.0, 1.0)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn windVisual(pos: vec2<f32>, wind: vec2<f32>, scale: f32, t: f32) -> vec3<f32> {
    let speed = length(wind);
    if speed < 0.1 { return vec3<f32>(0.0); }
    let dir = wind / speed; let across = vec2<f32>(-dir.y, dir.x);
    // Cat's paws: patches of gusts drifting downwind, longer along the wind than across it.
    let q = pos - dir * (scale / 14.0) * t;
    let stretched = vec2<f32>(dot(q, dir) / 2.2, dot(q, across)) / (scale / 6.0);
    var gust = 0.0; var amp = 0.55; var f = 1.0;
    for (var k = 0; k < 3; k++) { gust += amp * noise2(stretched * f + vec2<f32>(f32(k) * 17.0, 3.0)); amp *= 0.5; f *= 2.3; }
    gust = smoothstep(0.35, 0.8, gust);
    return vec3<f32>(0.0, gust, 0.0);
}
