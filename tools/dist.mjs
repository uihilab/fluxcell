// Build the minified, self-contained demo site in dist/ (not versioned; ready to upload to any static
// host): node tools/dist.mjs
//   dist/index.html, guide.html,     the pages (script and links rewritten for the layout below), their
//     references.html, site.css, site.js    shared style and script (theme, contents, code colours)
//   dist/js/app.js                    the demo app and the library, bundled and minified (esbuild)
//   dist/js/shaders/*.wgsl            solver and display shaders, comments stripped
//   dist/data/, dist/data/dem/        demo data and terrain tiles
// The bundle resolves shaders (./shaders/) and terrain tiles (../data/dem/) from its own URL, as the
// sources do, so the layout keeps those relative paths. esbuild is used at build time only, through
// npx at a pinned version; the library and the site have no runtime dependencies.
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ESBUILD = 'esbuild@0.24.2';
const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const out = join(root, 'dist');
const viewer = join(root, 'examples', 'viewer');
const tracked = prefix => execFileSync('git', ['ls-files', prefix], { cwd: root, encoding: 'utf8' }).split('\n').filter(Boolean);

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, 'js', 'shaders'), { recursive: true });

// 1. Bundle and minify the app (the dynamic node:fs import in src/terrain.mjs is for Node only).
execFileSync('npx', ['--yes', ESBUILD, join(viewer, 'main.mjs'), '--bundle', '--minify', '--format=esm', '--target=es2022',
    '--platform=browser', '--external:node:*', '--legal-comments=none', `--outfile=${join(out, 'js', 'app.js')}`],
    { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });

// 2. Shaders: comments, indentation and blank lines removed; //#include directives kept (WGSL has
//    no string literals, so everything after // on a line is a comment).
const shaderFiles = [...tracked('src/shaders'), ...tracked('examples/viewer/shaders')].filter(f => f.endsWith('.wgsl'));
for (const file of shaderFiles) {
    const lines = readFileSync(join(root, file), 'utf8').split(/\r?\n/).map(line => {
        if (/^\s*\/\/#include\s/.test(line)) return line.trim();
        return line.replace(/\/\/.*$/, '').trim();
    }).filter(Boolean);
    writeFileSync(join(out, 'js', 'shaders', file.split('/').pop()), lines.join('\n') + '\n');
}

// 3. Pages: the module script becomes the bundle.
const page = name => readFileSync(join(viewer, name), 'utf8');
writeFileSync(join(out, 'index.html'), page('index.html').replace('<script type="module" src="./main.mjs"></script>', '<script type="module" src="./js/app.js"></script>'));
for (const name of ['guide.html', 'references.html', 'site.css', 'site.js']) writeFileSync(join(out, name), page(name));
copyFileSync(join(viewer, 'logo_hilab.png'), join(out, 'logo_hilab.png'));
copyFileSync(join(root, 'LICENSE'), join(out, 'LICENSE'));

// 4. Data: the viewer's data next to the page, and the terrain tiles under data/dem/.
for (const file of [...tracked('examples/viewer/data'), ...tracked('data/dem')]) {
    const target = join(out, 'data', file.startsWith('data/dem/') ? join('dem', file.slice(9)) : file.slice('examples/viewer/data/'.length));
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(join(root, file), target);
}

writeFileSync(join(out, 'README.txt'), `FluxCell demos: a static site. Upload this folder as it is and open index.html.
Serve it over HTTPS (WebGPU needs a secure context) with .js as text/javascript.
Built by tools/dist.mjs from the repository (examples/viewer, src, data/dem). MIT licensed (LICENSE).
`);

const size = dir => readdirSync(dir).reduce((s, f) => { const p = join(dir, f), st = statSync(p); return s + (st.isDirectory() ? size(p) : st.size); }, 0);
const js = statSync(join(out, 'js', 'app.js')).size;
console.log(`dist: ${(size(out) / 1e6).toFixed(1)} MB in total; js/app.js ${(js / 1e3).toFixed(0)} kB; ${shaderFiles.length} shaders`);
