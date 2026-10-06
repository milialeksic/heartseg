"""Single-page web UI served at GET / by medseg.api. Plain HTML, CSS and JavaScript."""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>heartseg</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'
  viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='13' fill='none' stroke='%23eb6834'
  stroke-width='4'/%3E%3C/svg%3E">
<style>
  :root {
    color-scheme: dark;
    --bg: #0b0c0e;
    --panel: #14161a;
    --panel-2: #1b1e23;
    --border: #262a31;
    --text: #f2f3f5;
    --muted: #a1a8b2;
    --faint: #6f7681;
    --accent: #f07a45;
    --accent-strong: #eb6834;
    --accent-soft: rgba(235, 104, 52, 0.14);
    --ok: #3ccf91;
    --danger: #ff7a7a;
    --radius: 16px;
    --shadow: 0 1px 0 rgba(255, 255, 255, 0.03) inset, 0 16px 40px rgba(0, 0, 0, 0.35);
  }
  @media (prefers-color-scheme: light) {
    :root {
      color-scheme: light;
      --bg: #f3f4f6;
      --panel: #ffffff;
      --panel-2: #f7f8fa;
      --border: #e2e5ea;
      --text: #111318;
      --muted: #555c68;
      --faint: #8a919c;
      --accent: #e0612f;
      --accent-strong: #d4561f;
      --accent-soft: rgba(235, 104, 52, 0.1);
      --ok: #12a46b;
      --danger: #c62828;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 12px 32px rgba(16, 24, 40, 0.07);
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--bg); color: var(--text); line-height: 1.5;
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  button, input { font: inherit; color: inherit; }
  [hidden] { display: none !important; }

  /* Top bar */
  .topbar {
    position: sticky; top: 0; z-index: 10;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 14px 28px; border-bottom: 1px solid var(--border);
    background: color-mix(in srgb, var(--bg) 82%, transparent);
    backdrop-filter: blur(10px);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center;
    background: var(--accent-soft); color: var(--accent);
  }
  .name { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }
  .tagline { color: var(--muted); font-size: 0.82rem; }
  .pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
    border: 1px solid var(--border); border-radius: 999px; background: var(--panel);
    color: var(--muted); font-size: 0.82rem; white-space: nowrap;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
  .pill.ok .dot { background: var(--ok); box-shadow: 0 0 0 3px rgba(60, 207, 145, 0.18); }
  .pill.down .dot { background: var(--danger); }

  /* Layout */
  .layout {
    display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 24px;
    max-width: 1320px; margin: 0 auto; padding: 28px;
  }
  .side, .main { display: flex; flex-direction: column; gap: 20px; min-width: 0; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow);
  }
  .eyebrow {
    margin: 0 0 14px; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--faint);
  }
  .title { margin: 0; font-size: 1rem; font-weight: 600; }
  .faint { color: var(--faint); }
  .muted { color: var(--muted); }

  /* Upload */
  .drop {
    display: flex; flex-direction: column; align-items: center; gap: 6px; text-align: center;
    padding: 28px 16px; border: 1.5px dashed var(--border); border-radius: 12px;
    background: var(--panel-2); cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .drop:hover, .drop.over { border-color: var(--accent); background: var(--accent-soft); }
  .drop svg { color: var(--accent); margin-bottom: 4px; }
  .drop-title { font-weight: 600; }
  .drop-sub { color: var(--muted); font-size: 0.86rem; }
  .drop-sub u { color: var(--accent); text-decoration-thickness: 1px; }
  .chip {
    display: flex; align-items: center; gap: 10px; margin-top: 12px; padding: 10px 12px;
    border: 1px solid var(--border); border-radius: 10px; background: var(--panel-2);
    font-size: 0.9rem; min-width: 0;
  }
  .chip svg { color: var(--muted); flex: none; }
  #fileName { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    width: 100%; padding: 11px 16px; border-radius: 10px; border: 1px solid transparent;
    font-weight: 600; cursor: pointer; text-decoration: none;
    transition: background 0.15s, opacity 0.15s, transform 0.05s;
  }
  .btn:active { transform: translateY(1px); }
  .btn.primary { margin-top: 14px; background: var(--accent-strong); color: #fff; }
  .btn.primary:hover { background: var(--accent); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn.secondary {
    margin-top: 18px; background: var(--panel-2); border-color: var(--border);
    color: var(--text);
  }
  .btn.secondary:hover { border-color: var(--accent); }
  .progress {
    position: relative; height: 4px; margin-top: 14px; border-radius: 4px; overflow: hidden;
    background: var(--panel-2);
  }
  .progress div {
    position: absolute; inset: 0 auto 0 0; width: 35%; border-radius: 4px;
    background: var(--accent); animation: slide 1.1s ease-in-out infinite;
  }
  @keyframes slide { from { left: -35%; } to { left: 100%; } }
  .status { margin: 10px 0 0; min-height: 1.2em; font-size: 0.88rem; color: var(--muted); }
  .status.error { color: var(--danger); }

  /* Results */
  .hero { padding: 4px 0 16px; border-bottom: 1px solid var(--border); }
  .hero-label { color: var(--muted); font-size: 0.86rem; }
  .hero-value {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }
  .unit { font-size: 1rem; font-weight: 500; color: var(--muted); margin-left: 6px; }
  .kv { margin: 0; }
  .kv div {
    display: flex; justify-content: space-between; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid var(--border); font-size: 0.9rem;
  }
  .kv div:last-child { border-bottom: 0; }
  .kv dt { color: var(--muted); }
  .kv dd { margin: 0; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }
  .disclaimer { margin: 0; padding: 0 4px; font-size: 0.78rem; color: var(--faint); }

  /* Viewer */
  .viewer-bar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    flex-wrap: wrap; margin-bottom: 14px;
  }
  .controls { display: flex; align-items: center; gap: 8px; }
  .icon-btn {
    width: 36px; height: 36px; display: grid; place-items: center; cursor: pointer;
    border: 1px solid var(--border); border-radius: 10px; background: var(--panel-2);
  }
  .icon-btn:hover:not(:disabled) { border-color: var(--accent); }
  .icon-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .switch {
    display: inline-flex; align-items: center; gap: 10px; margin-right: 8px;
    font-size: 0.88rem; color: var(--muted); cursor: pointer; user-select: none;
  }
  .switch input { position: absolute; opacity: 0; width: 1px; height: 1px; }
  .track {
    width: 38px; height: 22px; border-radius: 999px; background: var(--border);
    position: relative; transition: background 0.15s;
  }
  .thumb {
    position: absolute; top: 3px; left: 3px; width: 16px; height: 16px; border-radius: 50%;
    background: #fff; transition: transform 0.15s;
  }
  .switch input:checked + .track { background: var(--accent-strong); }
  .switch input:checked + .track .thumb { transform: translateX(16px); }
  .switch input:focus-visible + .track { outline: 2px solid var(--accent); outline-offset: 2px; }
  .stage {
    position: relative; aspect-ratio: 1 / 1; max-height: 68vh; width: 100%;
    display: grid; place-items: center; overflow: hidden;
    background: #000; border-radius: 12px;
  }
  .stage img { width: 100%; height: 100%; object-fit: contain; }
  .empty { text-align: center; color: #8a909a; padding: 24px; max-width: 320px; }
  .empty svg { color: #3a3f47; margin-bottom: 10px; }
  .slider-row { display: flex; align-items: center; gap: 14px; margin-top: 16px; }
  input[type=range] { flex: 1; accent-color: var(--accent-strong); }
  .legend {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin-top: 12px; font-size: 0.84rem; color: var(--muted);
  }
  .swatch {
    width: 12px; height: 12px; border-radius: 3px; border: 2px solid var(--accent-strong);
    background: var(--accent-soft);
  }
  kbd {
    font-family: inherit; font-size: 0.78rem; padding: 1px 6px; border-radius: 5px;
    border: 1px solid var(--border); background: var(--panel-2); color: var(--muted);
  }
  .overview { display: block; width: 100%; height: auto; margin-top: 12px; border-radius: 12px; }

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; padding: 16px; }
    .topbar { padding: 12px 16px; }
    .tagline { display: none; }
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <div class="logo" aria-hidden="true">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12
          21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/>
        <path d="M3 12h4l2-3 3 6 2-3h7"/>
      </svg>
    </div>
    <div>
      <div class="name">heartseg</div>
      <div class="tagline">Left atrium segmentation &middot; 3D cardiac MRI</div>
    </div>
  </div>
  <div class="pill" id="health">
    <span class="dot"></span><span id="healthText">Connecting</span>
  </div>
</header>

<main class="layout">
  <aside class="side">
    <section class="panel">
      <h2 class="eyebrow">1 &middot; Upload</h2>
      <label id="drop" class="drop" for="file">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>
        </svg>
        <span class="drop-title">Drop a cardiac MRI volume</span>
        <span class="drop-sub">or <u>browse your files</u> &middot; .nii or .nii.gz</span>
        <input id="file" type="file" accept=".nii,.gz" hidden>
      </label>
      <div id="fileChip" class="chip" hidden>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.8" aria-hidden="true">
          <path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/>
          <path d="M14 3v5h5"/>
        </svg>
        <span id="fileName"></span><span id="fileSize" class="faint"></span>
      </div>
      <button id="run" class="btn primary" disabled>Segment</button>
      <div id="progress" class="progress" hidden><div></div></div>
      <p id="status" class="status" role="status" aria-live="polite"></p>
    </section>

    <section id="results" class="panel" hidden>
      <h2 class="eyebrow">2 &middot; Results</h2>
      <div class="hero">
        <div class="hero-label">Left atrium volume</div>
        <div class="hero-value"><span id="vol">-</span><span class="unit">mL</span></div>
      </div>
      <dl class="kv">
        <div><dt>Connected components</dt><dd id="comp"></dd></div>
        <div><dt>Image size (voxels)</dt><dd id="shape"></dd></div>
        <div><dt>Voxel spacing (mm)</dt><dd id="spacing"></dd></div>
        <div><dt>Processing time</dt><dd id="time"></dd></div>
        <div><dt>Model</dt><dd id="pipeline"></dd></div>
      </dl>
      <a id="download" class="btn secondary" download>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>
        </svg>
        Download mask (.nii.gz)
      </a>
    </section>

    <p class="disclaimer">Research prototype trained on 16 public MRI scans. Not a medical
      device and not for clinical decisions. Files are processed on this machine.</p>
  </aside>

  <section class="main">
    <div class="panel">
      <div class="viewer-bar">
        <div>
          <h2 class="title">Slice viewer</h2>
          <div class="faint" id="sliceInfo">No volume loaded</div>
        </div>
        <div class="controls">
          <label class="switch">
            <input id="overlay" type="checkbox" checked>
            <span class="track"><span class="thumb"></span></span>
            <span>Segmentation</span>
          </label>
          <button class="icon-btn" id="prev" aria-label="Previous slice" disabled>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>
          </button>
          <button class="icon-btn" id="next" aria-label="Next slice" disabled>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
          </button>
        </div>
      </div>
      <div class="stage">
        <div class="empty" id="empty">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            stroke-width="1.2" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="12" cy="12" r="4"/>
          </svg>
          <p>Upload a cardiac MRI volume to see the predicted left atrium here.</p>
        </div>
        <img id="slice" alt="Selected slice" hidden>
      </div>
      <div class="slider-row">
        <input id="slider" type="range" min="0" max="0" value="0" disabled aria-label="Slice">
      </div>
      <div class="legend">
        <span class="swatch"></span><span>Predicted left atrium</span>
        <span class="faint">&middot; scroll with <kbd>&larr;</kbd> <kbd>&rarr;</kbd></span>
      </div>
    </div>

    <div class="panel" id="overviewPanel" hidden>
      <h2 class="title">Overview</h2>
      <div class="faint">Three planes through the atrium and a projection of the whole
        prediction. Planes are labelled by image axis.</div>
      <img id="overview" class="overview" alt="Prediction in three planes and a projection">
    </div>
  </section>
</main>

<script>
  const $ = (id) => document.getElementById(id);
  const png = (b64) => "data:image/png;base64," + b64;
  const fileInput = $("file");
  let slices = [];
  let totalSlices = 0;
  let maskUrl = null;

  function setStatus(text, isError = false) {
    $("status").textContent = text;
    $("status").classList.toggle("error", isError);
  }

  function formatSize(bytes) {
    return bytes > 1e6 ? (bytes / 1e6).toFixed(1) + " MB" : Math.round(bytes / 1e3) + " kB";
  }

  async function checkHealth() {
    const pill = $("health");
    try {
      const r = await fetch("/health");
      const h = await r.json();
      pill.className = "pill ok";
      const n = h.models + (h.models === 1 ? " model" : " models");
      $("healthText").textContent = "Ready · " + n + " · " + h.device.toUpperCase();
    } catch {
      pill.className = "pill down";
      $("healthText").textContent = "Service unavailable";
    }
  }

  function chooseFile(file) {
    if (!file) return;
    if (!/\.nii(\.gz)?$/i.test(file.name)) {
      setStatus("Please choose a .nii or .nii.gz file.", true);
      $("run").disabled = true;
      return;
    }
    $("fileName").textContent = file.name;
    $("fileSize").textContent = formatSize(file.size);
    $("fileChip").hidden = false;
    $("run").disabled = false;
    setStatus("");
  }

  function showSlice() {
    const i = Number($("slider").value);
    const s = slices[i];
    if (!s) return;
    $("slice").src = png($("overlay").checked ? s.png : s.png_raw);
    $("sliceInfo").textContent =
      "Slice " + s.index + " of " + (totalSlices - 1) + " · view " + (i + 1) + "/" +
      slices.length;
    $("prev").disabled = i === 0;
    $("next").disabled = i === slices.length - 1;
  }

  function step(delta) {
    if (!slices.length) return;
    const v = Math.min(Math.max(Number($("slider").value) + delta, 0), slices.length - 1);
    $("slider").value = v;
    showSlice();
  }

  function render(d) {
    $("vol").textContent = d.volume_ml.toFixed(1);
    $("comp").textContent = d.n_components;
    $("shape").textContent = d.shape.join(" × ");
    $("spacing").textContent = d.spacing_mm.map((v) => v.toFixed(2)).join(" × ");
    $("time").textContent = d.elapsed_s.toFixed(1) + " s";
    $("pipeline").textContent =
      d.models + "-model ensemble" + (d.postprocess === "lcc" ? " + largest component" : "");
    $("overview").src = png(d.overview_png);

    slices = d.slices;
    totalSlices = d.n_slices_total;
    $("slider").max = Math.max(slices.length - 1, 0);
    $("slider").value = Math.floor(slices.length / 2);
    $("slider").disabled = slices.length < 2;
    $("empty").hidden = true;
    $("slice").hidden = false;
    showSlice();

    const bytes = Uint8Array.from(atob(d.mask_nii_gz), (c) => c.charCodeAt(0));
    if (maskUrl) URL.revokeObjectURL(maskUrl);
    maskUrl = URL.createObjectURL(new Blob([bytes], { type: "application/gzip" }));
    $("download").href = maskUrl;
    $("download").download = d.filename;

    $("results").hidden = false;
    $("overviewPanel").hidden = false;
  }

  async function segment() {
    const file = fileInput.files[0];
    if (!file) return;
    $("run").disabled = true;
    $("progress").hidden = false;
    setStatus("Segmenting " + file.name + "…");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch("/segment?output=preview", { method: "POST", body });
      const text = await response.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { detail: text }; }
      if (!response.ok) throw new Error(data.detail || response.statusText);
      render(data);
      setStatus("Done · " + file.name);
    } catch (err) {
      setStatus("Error: " + err.message, true);
    } finally {
      $("run").disabled = false;
      $("progress").hidden = true;
    }
  }

  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) =>
    drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) =>
    drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    chooseFile(file);
  });
  fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
  $("run").addEventListener("click", segment);
  $("slider").addEventListener("input", showSlice);
  $("overlay").addEventListener("change", showSlice);
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  document.addEventListener("keydown", (e) => {
    if (e.target === $("slider")) return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });

  checkHealth();
</script>
</body>
</html>
"""
