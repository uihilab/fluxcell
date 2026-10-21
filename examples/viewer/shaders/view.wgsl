// View parameters shared by the viewer's shaders (written by render.mjs: resize, draw).
struct View { width: u32, height: u32, canvasW: f32, canvasH: f32, zMin: f32, zMax: f32,
              depthScale: f32, wetDepth: f32, dx: f32, dy: f32, tracer: f32, mode: u32,
              logScale: u32, speedScale: f32, vorticityScale: f32, still: f32, elevationScale: f32,
              relief: f32, zoom: f32, centreX: f32, centreY: f32,
              time: f32, windX: f32, windY: f32, contour: f32, pad1: f32, pad2: f32, pad3: f32 }
