// Dependency-free static server for the browser tools (WebGPU needs localhost or HTTPS).
import { createServer } from 'node:http';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ROOT serves another folder (for example the dist copy: ROOT=dist npm run serve).
const root = resolve(process.env.ROOT ?? fileURLToPath(new URL('..', import.meta.url)));
const port = Number(process.env.PORT ?? 8080);
const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
    '.wgsl': 'text/plain', '.json': 'application/json', '.css': 'text/css', '.md': 'text/plain' };

createServer(async (request, response) => {
    let path = normalize(decodeURIComponent(new URL(request.url, 'http://x').pathname));
    let file = join(root, path);
    if (!file.startsWith(root)) { response.writeHead(403).end(); return; }
    // Local archiving: PUT /docs/results/<name>.json stores a run report, PUT /data/figures/<name>.json
    // stores figure fields (not versioned).
    if (request.method === 'PUT') {
        if (!/^[\\/](docs[\\/]results|data[\\/]figures)[\\/][\w.-]+\.json$/.test(path)) { response.writeHead(403).end(); return; }
        const chunks = [];
        for await (const chunk of request) chunks.push(chunk);
        const body = Buffer.concat(chunks).toString('utf8');
        try { JSON.parse(body); } catch { response.writeHead(400).end('Invalid JSON'); return; }
        await mkdir(dirname(file), { recursive: true });
        await writeFile(file, body);
        response.writeHead(201).end('Saved');
        return;
    }
    try {
        if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
        const body = await readFile(file);
        response.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream',
            'cache-control': 'no-store' });
        response.end(body);
    } catch { response.writeHead(404).end('Not found'); }
}).listen(port, '127.0.0.1', () => {
    console.log(`Serving ${root} at http://localhost:${port}/`);
    console.log(`  validation: http://localhost:${port}/tools/validate/`);
    console.log(`  benchmark:  http://localhost:${port}/tools/benchmark/`);
    console.log(`  demo:       http://localhost:${port}/examples/viewer/`);
});
