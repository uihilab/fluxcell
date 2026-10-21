// Independent exact solutions; four-point Gauss quadrature produces cell averages.
const points = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
const weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538];

export function simpleWave(x, time) {
    // g=1, c0=1, amplitude=0.03; right-going invariant u-2c=-2.
    let foot = x - time;
    for (let i = 0; i < 12; i++) {
        const a = 2 * Math.PI * foot;
        foot -= (foot + (1 + 0.09 * Math.sin(a)) * time - x) / (1 + 0.18 * Math.PI * Math.cos(a) * time);
    }
    const c = 1 + 0.03 * Math.sin(2 * Math.PI * foot), h = c * c, u = 2 * (c - 1);
    return [h, h * u, 0];
}

export function damSolution(rightDepth, gravity = 9.81, leftDepth = 2, origin = 50) {
    const cl = Math.sqrt(gravity * leftDepth);
    let hs = 0, us = 0, shock = 0;
    if (rightDepth > 0) {
        const waveCurve = (h, hk) => h > hk ? (h - hk) * Math.sqrt(gravity / 2 * (1 / h + 1 / hk))
            : 2 * (Math.sqrt(gravity * h) - Math.sqrt(gravity * hk));
        let low = rightDepth, high = leftDepth;
        for (let i = 0; i < 80; i++) {
            hs = (low + high) / 2;
            if (waveCurve(hs, leftDepth) + waveCurve(hs, rightDepth) > 0) high = hs; else low = hs;
        }
        us = -waveCurve(hs, leftDepth);
        shock = hs * us / (hs - rightDepth);
    }
    return (x, time) => {
        if (time === 0) return [x < origin ? leftDepth : rightDepth, 0, 0];
        const s = (x - origin) / time;
        if (s <= -cl) return [leftDepth, 0, 0];
        const tail = rightDepth === 0 ? 2 * cl : us - Math.sqrt(gravity * hs);
        if (s < tail) { const h = ((2 * cl - s) / 3) ** 2 / gravity; return [h, h * 2 * (cl + s) / 3, 0]; }
        if (rightDepth === 0) return [0, 0, 0];
        if (s < shock) return [hs, hs * us, 0];
        return [rightDepth, 0, 0];
    };
}

export function cellAverages(width, dx, sample, time = 0) {
    const state = new Float64Array(4 * width);
    for (let i = 0; i < width; i++) for (let j = 0; j < 4; j++) {
        const value = sample((i + 0.5 + points[j] / 2) * dx, time);
        for (let k = 0; k < 3; k++) state[4 * i + k] += weights[j] / 2 * value[k];
    }
    return state;
}

export function exactScenario(name, width) {
    const smooth = name === 'simple-wave';
    if (!smooth && !['wet-dam', 'dry-dam'].includes(name)) throw new Error(`Unknown exact case ${name}`);
    const dx = (smooth ? 1 : 100) / width;
    const sample = smooth ? simpleWave : damSolution(name === 'wet-dam' ? 0.25 : 0);
    const config = { width, height: 1, dx, dy: smooth ? 1 : 100, gravity: smooth ? 1 : 9.81,
        boundaryX: smooth ? 'periodic' : 'wall', boundaryY: 'periodic' };
    const finalTime = smooth ? 0.1 : 2;
    return { config, finalTime, sample,
        initial: { state: cellAverages(width, dx, sample), bed: new Float64Array(width) } };
}

export function steadyBedScenario(width) {
    const q = 0.2, energy = 1.02, dx = 1 / width;
    const sample = x => {
        const z = 0.1 * Math.sin(2 * Math.PI * x);
        let h = energy - z;
        for (let i = 0; i < 12; i++) h -= (h + z + q * q / (2 * h * h) - energy) / (1 - q * q / (h * h * h));
        return [h, q, 0];
    };
    const bed = Float64Array.from({ length: width }, (_, i) =>
        0.1 * Math.sin(2 * Math.PI * (i + 0.5) * dx) * Math.sin(Math.PI * dx) / (Math.PI * dx));
    return { config: { width, height: 1, dx, dy: 1, gravity: 1, boundaryX: 'periodic', boundaryY: 'periodic' },
        initial: { state: cellAverages(width, dx, sample), bed }, finalTime: 0.1, sample };
}

export function diagonalAverages(width, time) {
    const state = new Float64Array(4 * width * width);
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
            const phase = (x + y + 1 + (points[a] + points[b]) / 2) / width;
            const value = simpleWave(phase, Math.SQRT2 * time), weight = weights[a] * weights[b] / 4;
            const k = 4 * (y * width + x);
            state[k] += weight * value[0];
            state[k + 1] += weight * value[1] / Math.SQRT2;
            state[k + 2] += weight * value[1] / Math.SQRT2;
        }
    }
    return state;
}

export function diagonalScenario(width) {
    return { config: { width, height: width, dx: 1 / width, dy: 1 / width, gravity: 1,
        boundaryX: 'periodic', boundaryY: 'periodic' }, finalTime: 0.05,
        initial: { state: diagonalAverages(width, 0), bed: new Float64Array(width * width) } };
}

export function depthErrors(state, exact) {
    let l1 = 0, l2 = 0, linf = 0;
    for (let i = 0; i < state.length; i += 4) {
        const error = Math.abs(state[i] - exact[i]);
        l1 += error; l2 += error * error; linf = Math.max(linf, error);
    }
    const n = state.length / 4;
    return { l1: l1 / n, l2: Math.sqrt(l2 / n), linf };
}
