// Duplicates page — talks to the duplicates blueprint at DUP_API.
const $ = (s) => document.querySelector(s);
const fmtSize = (n) => { const u = ['B','KB','MB','GB','TB']; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };
const fmtDate = (t) => new Date(t * 1000).toLocaleString();
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

let pairs = [], root = null, scope = 'pairs', filter = 'all', shown = 100, imgThreshold = 0.92, polling = null, jobId = null;

const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
$('#root').value = store.get('duplicates.root', '');

$('#nameT').oninput = (e) => $('#nameTv').textContent = e.target.value + '%';
$('#imgT').oninput = (e) => { imgThreshold = e.target.value / 100; $('#imgTv').textContent = imgThreshold.toFixed(2); render(); };
for (const seg of ['#scope', '#filter']) {
  $(seg).onclick = (e) => {
    const b = e.target.closest('button'); if (!b) return;
    $(seg).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    if (seg === '#scope') scope = b.dataset.v; else { filter = b.dataset.v; shown = 100; render(); }
  };
}
$('#search').oninput = () => { shown = 100; render(); };
$('#sort').onchange = render;
$('#more').onclick = () => { shown += 100; render(); };

$('#browse').onclick = async () => {
  $('#browse').disabled = true;
  try { const r = await api('/api/pick-folder', { start: $('#root').value });
        if (r.path) $('#root').value = r.path; }
  catch (e) { showErr(e.message); }
  finally { $('#browse').disabled = false; }
};

function showErr(m) { $('#err').textContent = m || ''; $('#err').classList.toggle('on', !!m); }
function setBusy(b) {
  ['#runL1', '#runBytes', '#runImg', '#browse'].forEach((s) => $(s).disabled = b);
  if (!b) { $('#runBytes').disabled = !root; $('#runImg').disabled = !root || !imageOk; }
  $('#progress').classList.toggle('on', b);
}

async function runJob(url, body) {
  showErr('');
  try {
    const job = await api(url, body);
    jobId = job.id; setBusy(true); poll();
  } catch (e) { showErr(e.message); }
}
function poll() {
  clearTimeout(polling);
  polling = setTimeout(async () => {
    const j = await api(DUP_API + '/job/' + jobId).catch((e) => ({ status: 'error', error: e.message }));
    const bar = $('#bar');
    bar.classList.toggle('indet', !j.total);
    if (j.total) bar.firstElementChild.style.width = Math.round(100 * j.done / j.total) + '%';
    $('#msg').textContent = `${j.message || ''}  (${j.elapsed ?? 0}s)`;
    if (j.status === 'running') return poll();
    setBusy(false);
    if (j.status === 'error') showErr(j.error);
    await loadPairs();
    if (j.status === 'done') $('#count').textContent += ` · last step: ${j.message}`;
  }, 400);
}
$('#cancel').onclick = () => jobId && api(`${DUP_API}/job/${jobId}/cancel`, {});

$('#runL1').onclick = () => {
  const r = $('#root').value.trim();
  if (!r) return showErr('Choose a folder first.');
  store.set('duplicates.root', r);
  runJob(DUP_API + '/scan/level1', {
    root: r, name_threshold: $('#nameT').value / 100, size_tolerance_pct: +$('#sizeTol').value,
    include_hidden: $('#hidden').checked, skip_empty: $('#skipEmpty').checked,
    min_size: Math.round(+$('#minSize').value * 1024) });
};
$('#runBytes').onclick = () => runJob(DUP_API + '/scan/bytes', { scope });
$('#runImg').onclick = () => runJob(DUP_API + '/scan/images', { scope, model: $('#model').value, threshold: imgThreshold });

async function loadPairs() {
  const r = await api(DUP_API + '/pairs');
  pairs = r.pairs; root = r.root; render();
}

function matches(p) {
  const q = $('#search').value.trim().toLowerCase();
  if (q && !(p.a.path.toLowerCase().includes(q) || p.b.path.toLowerCase().includes(q))) return false;
  switch (filter) {
    case 'bytes': return p.bytes_equal === true;
    case 'different': return p.bytes_equal === false;
    case 'image': return (p.image_sim ?? 0) >= imgThreshold;
    case 'unverified': return p.bytes_equal == null && p.image_sim == null;
  }
  return true;
}
const sorters = {
  size: (a, b) => Math.max(b.a.size, b.b.size) - Math.max(a.a.size, a.b.size),
  name: (a, b) => b.name_score - a.name_score,
  image: (a, b) => (b.image_sim ?? -1) - (a.image_sim ?? -1),
  path: (a, b) => a.a.path.localeCompare(b.a.path),
};

function relPath(p) {
  const rel = root && p.startsWith(root + '/') ? p.slice(root.length + 1) : p;
  const i = rel.lastIndexOf('/');
  const dir = root && p.startsWith(root + '/') ? esc(root) + '/' : '';
  return dir + (i >= 0 ? esc(rel.slice(0, i + 1)) : '') + '<b>' + esc(rel.slice(i + 1)) + '</b>';
}

function sideHtml(f, label, key) {
  const t = f.is_image ? `<img class="thumb" loading="lazy" src="${DUP_API}/thumb?path=${encodeURIComponent(f.path)}" alt="">` : '';
  return `<div class="side"><span class="tag">${label}</span>${t}
    <div class="path" title="${esc(f.path)}">${relPath(f.path)}</div>
    <div class="meta">${fmtSize(f.size)} · modified ${fmtDate(f.mtime)}</div>
    <div class="actions">
      <button data-act="reveal" data-path="${esc(f.path)}">Show in Finder</button>
      ${imageActions(f.path, { forget: f.indexed })}
    </div></div>`;
}

function badges(p) {
  const out = [];
  const n = Math.round(p.name_score * 100);
  out.push(`<span class="badge ${n >= 90 ? 'good' : ''}">Name ${n}%</span>`);
  out.push(p.size_match ? '<span class="badge good">Same size</span>'
                        : `<span class="badge warn">Size Δ ${fmtSize(Math.abs(p.a.size - p.b.size))}</span>`);
  if (p.bytes_equal === true) out.push('<span class="badge good">✓ Byte-identical</span>');
  else if (p.bytes_equal === false) out.push('<span class="badge bad">✗ Bytes differ</span>');
  if (p.image_sim != null) {
    const ok = p.image_sim >= imgThreshold;
    out.push(`<span class="badge ${ok ? 'good' : 'warn'}">Image ${p.image_sim.toFixed(3)} · ${p.image_model.toUpperCase()}</span>`);
  }
  if (p.error) out.push(`<span class="badge bad" title="${esc(p.error)}">Read error</span>`);
  out.push(`<span class="badge">found by: ${p.found_by.join(', ')}</span>`);
  return out.join('');
}

function render() {
  const list = pairs.filter(matches).sort(sorters[$('#sort').value]);
  $('#count').textContent = `${list.length.toLocaleString()} of ${pairs.length.toLocaleString()} pairs`;
  $('#more').style.display = list.length > shown ? '' : 'none';
  if (!pairs.length) {
    $('#list').innerHTML = `<div class="empty">${root ? 'No duplicate candidates found with these settings.' : 'Choose a folder and run Level 1 to get started.'}</div>`;
  } else if (!list.length) {
    $('#list').innerHTML = '<div class="empty">Nothing matches this filter.</div>';
  } else {
    $('#list').innerHTML = list.slice(0, shown).map((p) =>
      `<div class="pair"><div class="pair-head">${badges(p)}</div>
        <div class="sides">${sideHtml(p.a, 'File A')}${sideHtml(p.b, 'File B')}</div></div>`).join('');
  }
  if (!busy()) { $('#runBytes').disabled = !root; $('#runImg').disabled = !root || !imageOk; }
}
const busy = () => $('#progress').classList.contains('on');

$('#list').onclick = async (e) => {
  const b = e.target.closest('button[data-act],button[data-img-act]'); if (!b) return;
  const path = b.dataset.path;
  if (b.dataset.act === 'reveal') return api(DUP_API + '/reveal', { path }).catch((x) => showErr(x.message));
  const forget = b.dataset.imgAct === 'forget';
  b.disabled = true;
  try {
    const done = forget ? await forgetImage(path, DUP_API + '/forget') : await deleteImage(path, DUP_API + '/delete');
    if (!done) { b.disabled = false; return; }
    const name = path.split('/').pop();
    if (forget) {
      // file stays in the results; it just isn't searchable any more
      pairs.forEach((p) => { [p.a, p.b].forEach((f) => { if (f.path === path) f.indexed = false; }); });
      toast(`Forgot ${name} from the index.`);
    } else {
      pairs = pairs.filter((p) => p.a.path !== path && p.b.path !== path);
      toast(done === 'permanent' ? `Permanently deleted ${name}.` : `Moved ${name} to the Trash.`);
    }
    render();
  } catch (x) { showErr(x.message); b.disabled = false; }
};

const imageOk = window.IMAGE_OK;
(async () => {
  const s = await api(DUP_API + '/status');
  if (s.root) await loadPairs();
})();
