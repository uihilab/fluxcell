// Shared page script, loaded in <head> as a plain script. Light or dark pages: light unless the reader
// has chosen dark with the header button (#theme), remembered in this browser and set before the first
// paint. On the guide and references: the section in view is marked in the contents, and the code
// examples are coloured (comments, strings, keywords, numbers).
(() => {
    const KEY = 'fluxcell-theme', root = document.documentElement;
    let theme = 'light';
    try { if (localStorage.getItem(KEY) === 'dark') theme = 'dark'; } catch { /* storage blocked: light */ }
    root.dataset.theme = theme;
    addEventListener('DOMContentLoaded', () => {
        const button = document.getElementById('theme');
        if (button) {
            const label = () => { button.textContent = root.dataset.theme === 'dark' ? 'Light' : 'Dark'; };
            label();
            button.addEventListener('click', () => {
                root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
                try { localStorage.setItem(KEY, root.dataset.theme); } catch { /* not remembered */ }
                label();
            });
        }
        for (const block of document.querySelectorAll('pre code')) block.innerHTML = colour(block.textContent);
        // Contents: the link of the topmost section in the scrolling pane is marked.
        const links = [...document.querySelectorAll('.contents a[href^="#"]')], pane = document.querySelector('.scroll');
        if (!links.length || !pane) return;
        const sections = links.map(a => document.getElementById(a.getAttribute('href').slice(1)));
        const mark = () => {
            const top = pane.getBoundingClientRect().top + 24;
            let current = 0;
            sections.forEach((s, k) => { if (s && s.getBoundingClientRect().top <= top) current = k; });
            if (pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 2) current = sections.length - 1;
            links.forEach((a, k) => a.setAttribute('aria-current', String(k === current)));
        };
        pane.addEventListener('scroll', mark, { passive: true });
        mark();
    });

    // JavaScript tokens, in order of precedence; everything else is left as it is.
    const TOKENS = /(\/\/[^\n]*)|('(?:[^'\\\n]|\\.)*'|`[^`]*`|"(?:[^"\\\n]|\\.)*")|\b(import|from|export|const|let|var|await|async|new|for|of|in|if|else|return|try|finally|catch|function|while|true|false|null|undefined)\b|\b(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b/g;
    const KINDS = ['comment', 'string', 'keyword', 'number'];
    const escape = text => text.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
    function colour(code) {
        let html = '', last = 0;
        for (const m of code.matchAll(TOKENS)) {
            const kind = KINDS[m.slice(1).findIndex(g => g !== undefined)];
            html += escape(code.slice(last, m.index)) + `<span class="${kind}">${escape(m[0])}</span>`;
            last = m.index + m[0].length;
        }
        return html + escape(code.slice(last));
    }
})();
