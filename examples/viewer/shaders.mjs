// Loads the viewer's WGSL from shaders/. A line `//#include name.wgsl` is replaced by that file
// (each file at most once per shader), so the View struct, grid indexing, colormaps and the wind
// visual are written once and shared.
const cache = new Map();
const read = name => {
    if (!cache.has(name)) cache.set(name, fetch(new URL(`./shaders/${name}`, import.meta.url)).then(r => {
        if (!r.ok) throw new Error(`Cannot load shader ${name} (HTTP ${r.status})`);
        return r.text();
    }));
    return cache.get(name);
};
async function expand(name, seen) {
    if (seen.has(name)) return '';
    seen.add(name);
    const lines = (await read(name)).split('\n');
    const out = [];
    for (const line of lines) {
        const m = line.match(/^\/\/#include\s+(\S+)\s*$/);
        out.push(m ? await expand(m[1], seen) : line);
    }
    return out.join('\n');
}
// Resolves to { name: code } for the given shader names (without the .wgsl extension).
export async function loadShaders(names) {
    const entries = await Promise.all(names.map(async n => [n, await expand(`${n}.wgsl`, new Set())]));
    return Object.fromEntries(entries);
}
