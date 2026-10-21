// Dam breaks and spills: Oroville (2017), Teton Dam (1976), the Fennema-Chaudhry partial breach and a
// contaminated city dam break.

// Each entry returns { config, initial, ... } (fields in scenarios/index.mjs).
export const scenarios = {
    // Oroville Dam spillways at the lake's peak, 03:00 on 12 February 2017 (tools/dem/oroville_grid.py):
    // the lake held at 902.59 ft (CDEC ORO) overtops the emergency spillway (crest 901 ft), while the
    // main spillway releases 55,000 ft³/s into its chute. ?check=emergency: no release, walls on the
    // edges away from the lake, and a gauge on the spill side whose volume gain is the emergency
    // spillway discharge.
    async oroville({ release }) {
        const check = new URLSearchParams(location.search).get('check') === 'emergency';
        const [meta, bedBytes, maskBytes] = await Promise.all([fetch('./data/oroville.json').then(r => r.json()),
            ...['bed.f32', 'mask.u8'].map(f => fetch(`./data/oroville-${f}`).then(r => r.arrayBuffer()))]);
        const { width: W, height: H, dx } = meta, n = W * H, bed = new Float32Array(bedBytes), mask = new Uint8Array(maskBytes);
        const state = new Float32Array(4 * n), solid = new Uint8Array(n), source = new Float32Array(n), level = meta.peak_NAVD88_m;
        const releaseCells = mask.reduce((a, v) => a + (v === 2), 0);
        for (let i = 0; i < n; i++) {
            if (mask[i] === 3) solid[i] = 1;
            if (mask[i] === 1) state[4 * i] = Math.max(0, level - bed[i]);
            if (mask[i] === 2 && !check) source[i] = release * 1000 * 0.3048 ** 3 / (releaseCells * dx * dx);
        }
        // Stage segments (the lake beyond the window) wherever lake cells meet an edge.
        const hydrographs = [];
        const edge = (name, length, at) => {
            for (let k = 0; k < length;) {
                if (mask[at(k)] !== 1) { k++; continue; }
                let e = k; while (e < length && mask[at(e)] === 1) e++;
                hydrographs.push({ edge: name, kind: 'stage', start: k, end: e, series: [[0, level]] }); k = e;
            }
        };
        edge('west', H, k => k * W); edge('east', H, k => k * W + W - 1); edge('south', W, k => k); edge('north', W, k => (H - 1) * W + k);
        const releaseSource = kcfs => { const out = new Float32Array(n); for (let i = 0; i < n; i++) if (mask[i] === 2 && !check) out[i] = kcfs * 1000 * 0.3048 ** 3 / (releaseCells * dx * dx); return out; };
        source.set(releaseSource(release));
        const boundary = check ? 'wall' : 'outflow';
        return { config: { width: W, height: H, dx, dy: dx, boundaryX: boundary, boundaryY: boundary, hydrographs },
            initial: { state, bed, manning: new Float32Array(n).fill(0.016), source, solid }, releaseSource,
            depthScale: 3, wetDepth: 0.01, rate: 6, colour: 'speed', speedScale: 25, view3d: true, exaggeration: 1.5, elevationDatum: 'm NAVD88',
            camera: { target: [-350, 100, 0], distance: 1700, yaw: -2.45, pitch: 0.55 },
            ...(check ? { gauge: { mask: Uint8Array.from(mask, v => v !== 1 && v !== 3 ? 1 : 0), label: 'Spill side', sign: -1, reference: 'reported emergency spillway flow about 12,600 ft³/s' } } : {}),
            note: `Oroville Dam, California, at 03:00 on 12 February 2017: Lake Oroville at its peak, 902.59 ft (CDEC station ORO), stands 0.48 m over the 901 ft crest of the emergency spillway, which overflows for the first time, while the main spillway releases 55,000 ft³/s (the recorded outflow) down its chute. Today's 3DEP terrain (2 m), with the rebuilt spillways; the main spillway's gates are closed in the model and its release enters below them. Over today's crest the model passes about 230 m³/s (8,100 ft³/s) over the emergency spillway; DWR reported a peak of 12,600 ft³/s in 2017, over the original crest and eroding ground.` };
    },
    // Sediment plume: a river (400 m³/s, 1 kg/m³ of suspended mud) enters a 4 m deep basin through a
    // 150 m channel and leaves over the east edge. The mud flocculates and settles at
    // w_s = k c^(4/3) (Mehta, 1986), k = 1e-3 m/s (kg/m³)^(-4/3): 1 mm/s at 1 kg/m³ (illustrative).
    // Teton Dam, Idaho, 5 June 1976 (tools/dem/teton_grid.py): the reservoir at its level at failure
    // (5,301.7 ft, USBR; NGVD29 -> NAVD88 +1.10 m) behind the dam axis, released at once over today's
    // 3DEP terrain (40 m). The readout tracks the reservoir volume and the outflow at the dam
    // (-dV/dt) for comparison with the USGS flood surge of 2.3 million ft³/s.
    async 'teton-dam'() {
        const [meta, bedBytes, maskBytes] = await Promise.all([fetch('./data/teton.json').then(r => r.json()),
            ...['bed.f32', 'reservoir.u8'].map(f => fetch(`./data/teton-${f}`).then(r => r.arrayBuffer()))]);
        const { width: W, height: H, dx } = meta, n = W * H, bed = new Float32Array(bedBytes), mask = new Uint8Array(maskBytes);
        const state = new Float32Array(4 * n), level = meta.levelAtFailure_NAVD88_m;
        for (let i = 0; i < n; i++) if (mask[i]) state[4 * i] = Math.max(0, level - bed[i]);
        return { config: { width: W, height: H, dx, dy: dx, boundaryX: 'outflow', boundaryY: 'outflow' },
            initial: { state, bed, manning: new Float32Array(n).fill(0.035) },
            depthScale: 20, wetDepth: 0.05, rate: 90, colour: 'depth', view3d: true, exaggeration: 3, elevationDatum: 'm NAVD88',
            camera: { target: [-6500, 0, 0], distance: 15000, yaw: -1.3, pitch: 0.62 },
            gauge: { mask, label: 'Reservoir', reference: 'USGS: 2.3 million ft³/s (65,000 m³/s)' },
            note: `Teton Dam, Idaho, failed on 5 June 1976 with the reservoir at 5,301.7 ft (Bureau of Reclamation). Its ${(meta.reservoirVolume_m3 / 1e6).toFixed(0)} million m³ on today's 3DEP terrain (40 m cells; USGS: about 240,000 acre-ft, 296 million m³, released) are let go at once and run out of the canyon onto the Snake River Plain (the model ends 10 km past the dam). The readout gives the outflow at the dam; the USGS estimated a flood surge of 2.3 million ft³/s (65,000 m³/s). The real breach opened over about an hour, so this instant failure releases a higher peak; uniform Manning n = 0.035 is also an idealization.` };
    },
    // Partial breach (Fennema and Chaudhry 1990): 200 m x 200 m flat basin, 10 m of water behind
    // a 5 m thick dam with an asymmetric 75 m breach, 5 m of tailwater, no friction.
    'partial-breach'({ tailwater = 5, n0 = 0 } = {}) {
        const W = 400, H = 400, d = 0.5, n = W * H;
        const bed = new Float32Array(n), state = new Float32Array(4 * n), solid = new Uint8Array(n);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, X = (x + 0.5) * d, Y = (y + 0.5) * d;
            solid[i] = X >= 95 && X < 100 && !(Y >= 95 && Y < 170) ? 1 : 0;
            if (!solid[i]) state[4 * i] = X < 95 ? 10 : tailwater;
        }
        return { config: { width: W, height: H, dx: d, dy: d, boundaryX: 'outflow', boundaryY: 'wall' },
            initial: { state, bed, manning: new Float32Array(n).fill(n0), solid }, depthScale: 10, wetDepth: 0.02, rate: 1,
            note: 'Fennema and Chaudhry (1990): 10 m of water behind a dam fails over an asymmetric 75 m gap into 5 m of tailwater; the bore and the reflected waves are resolved without oscillations.' };
    },
    // Base of the contaminated urban dam break (not a menu entry).
    urban(W = 512, H = 256) {
        const n = W * H, bed = new Float32Array(n), state = new Float32Array(4 * n);
        const manning = new Float32Array(n).fill(0.02), solid = new Uint8Array(n);
        const level = 0.004 * (W - 90) + 4;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x;
            bed[i] = 0.004 * (W - x);
            const bx = (x - 140) % 26, by = (y + (Math.floor((x - 140) / 26) % 2) * 13) % 26;
            const building = x >= 140 && x < W - 30 && bx < 18 && by < 18 && (Math.floor(x / 26) * 7 + Math.floor(y / 26) * 3) % 5 !== 0;
            solid[i] = x === 0 || building ? 1 : 0;
            if (!solid[i] && x < 90) state[4 * i] = Math.max(0, level - bed[i]);
        }
        return { config: { width: W, height: H, dx: 1, dy: 1, boundaryX: 'open', boundaryY: 'wall' },
            initial: { state, bed, manning, solid }, depthScale: 4, wetDepth: 0.03, rate: 30 };
    },
    // Urban dam break whose reservoir holds a contaminated patch.
    'urban-spill'() {
        const s = scenarios.urban(), { width: W, height: H } = s.config, concentration = new Float32Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < 90; x++) if ((x - 55) ** 2 + (y - 170) ** 2 < 30 ** 2) concentration[y * W + x] = 1;
        return { ...s, config: { ...s.config, tracer: true }, initial: { ...s.initial, concentration }, concentration: 'log',
            view3d: true, exaggeration: 3, buildingHeight: 4,
            note: 'A reservoir holding 4 m of water fails onto a sloping grid of city blocks (synthetic, 1 m cells); a contaminated patch in the reservoir (c = 1) is carried between the buildings and diluted by the flood.' };
    },
};
