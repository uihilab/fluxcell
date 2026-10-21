// GPU pipelines of the viewer, created once per device. Per-demo buffers and bind groups are made in
// resources.mjs; the passes are recorded in draw.mjs.
import { loadShaders } from '../shaders.mjs';

const VIEW_BYTES = 112, SCENE_BYTES = 112;
const blendOver = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
const entries = (visibility, types) => types.map((type, binding) => ({ binding, visibility, buffer: { type } }));

export async function createRenderer(device, context, format) {
    const shaders = await loadShaders(['map2d', 'waterline', 'scene3d', 'particles-advect', 'particles-draw', 'bed', 'cylinders']);
    const module = name => device.createShaderModule({ code: shaders[name] });
    const fragment = GPUShaderStage.FRAGMENT, compute = GPUShaderStage.COMPUTE, both = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const layoutOf = bindGroupLayout => device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // 2D map: one full-screen triangle (map2d.wgsl).
    const map = module('map2d');
    const layout = device.createBindGroupLayout({ entries: entries(fragment, ['uniform', ...Array(8).fill('read-only-storage')]) });
    const pipeline = device.createRenderPipeline({ layout: layoutOf(layout),
        vertex: { module: map, entryPoint: 'vs' }, fragment: { module: map, entryPoint: 'fs', targets: [{ format }] } });

    // Waterline, water level and slope fields, computed per cell before each frame (waterline.wgsl).
    const waterline = module('waterline');
    const waterlineLayout = device.createBindGroupLayout({ entries: entries(compute,
        ['uniform', 'read-only-storage', 'read-only-storage', 'storage', 'storage', 'storage', 'storage']) });
    const waterlinePipelines = ['signed', 'blur'].map(entryPoint => device.createComputePipeline({ layout: layoutOf(waterlineLayout),
        compute: { module: waterline, entryPoint } }));

    // 3D view: ground and water meshes (scene3d.wgsl).
    const scene = module('scene3d');
    const sceneLayout = device.createBindGroupLayout({ entries: entries(both,
        ['uniform', ...Array(5).fill('read-only-storage'), 'uniform', 'read-only-storage']) });
    const scenePipeline = (vs, fs, blend) => device.createRenderPipeline({ layout: layoutOf(sceneLayout),
        vertex: { module: scene, entryPoint: vs },
        fragment: { module: scene, entryPoint: fs, targets: [{ format, ...(blend ? { blend: blendOver } : {}) }] },
        primitive: { topology: 'triangle-list' },
        // The water surface is drawn with a slope-scaled depth bias toward the camera: films a few
        // millimetres thick on steep rock would otherwise fight the terrain for depth and flicker.
        depthStencil: { format: 'depth24plus', depthWriteEnabled: !blend, depthCompare: 'less',
            ...(blend ? { depthBias: -4, depthBiasSlopeScale: -4 } : {}) } });

    // Flow tracers: moved by a compute pass, drawn as streaks in 2D or 3D (particles-*.wgsl).
    const advect = module('particles-advect'), streaks = module('particles-draw');
    const tracerComputeLayout = device.createBindGroupLayout({ entries: entries(compute,
        ['uniform', 'read-only-storage', 'read-only-storage', 'storage', 'uniform']) });
    const tracerRenderLayout = device.createBindGroupLayout({ entries: entries(both,
        ['uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'uniform', 'uniform']) });
    const streakPipeline = (vs, depth) => device.createRenderPipeline({ layout: layoutOf(tracerRenderLayout),
        vertex: { module: streaks, entryPoint: vs }, fragment: { module: streaks, entryPoint: 'fs', targets: [{ format, blend: blendOver }] },
        ...(depth ? { depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' } } : {}) });
    const tracers = { computeLayout: tracerComputeLayout, renderLayout: tracerRenderLayout,
        advect: device.createComputePipeline({ layout: layoutOf(tracerComputeLayout), compute: { module: advect, entryPoint: 'advect' } }),
        draw2d: streakPipeline('vs2d', false), draw3d: streakPipeline('vs3d', true),
        params: device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), frame: 0, on: false };

    // Bed evolution for the scour demo (bed.wgsl).
    const bed = module('bed');
    const morphLayout = device.createBindGroupLayout({ entries: entries(compute, ['uniform', 'read-only-storage', 'storage', 'storage', 'uniform']) });
    const morphPipeline = entryPoint => device.createComputePipeline({ layout: layoutOf(morphLayout), compute: { module: bed, entryPoint } });
    const morph = { layout: morphLayout, fluxes: morphPipeline('fluxes'), exner: morphPipeline('exner'), slides: morphPipeline('slides'),
        settle: morphPipeline('settle'), params: device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), on: false };

    // Round piles and islands as smooth cylinders in 3D (cylinders.wgsl).
    const cylinder = module('cylinders');
    const cylinderLayout = device.createBindGroupLayout({ entries: entries(both, ['uniform', 'read-only-storage']) });
    const cylinders = { layout: cylinderLayout, count: 0,
        pipeline: device.createRenderPipeline({ layout: layoutOf(cylinderLayout),
            vertex: { module: cylinder, entryPoint: 'vs' }, fragment: { module: cylinder, entryPoint: 'fs', targets: [{ format }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' } }) };

    return { context, layout, pipeline, waterlineLayout, waterlinePipelines, sceneLayout, tracers, morph, cylinders,
        groundPipeline: scenePipeline('vsGround', 'fsGround', false), waterPipeline: scenePipeline('vsWater', 'fsWater', true),
        sceneUniform: device.createBuffer({ size: SCENE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        view: device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) };
}
