/* Video stabilization motion-analysis player - vanilla JS, no CDN. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  samples: [],
  jobId: null,
  job: null,
  plan: null, // {frames, metrics, params}
  loadedVersion: null,
  trajCh: "x",
  frame: 0,
  hover: null,
  renderQueued: false,
  lastDrawnFrame: -1,
};

async function api(url, opts) {
  const r = await fetch(url, opts);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((body && body.error) || `HTTP ${r.status}`);
  return body;
}

/* ----------------------------------------------------------- samples */

async function loadSamples() {
  state.samples = await api("/api/samples");
  const box = $("sampleCards");
  box.innerHTML = "";
  state.samples.forEach((s) => {
    const b = document.createElement("button");
    b.className = "sample-chip";
    b.dataset.key = s.key;
    b.innerHTML = `<b>${s.title}</b><span>${s.desc}</span>`;
    b.onclick = () => loadSample(s.key, b);
    box.appendChild(b);
  });
  // auto-load the ground-truth planar sample for an instant demo
  const first = box.querySelector(".sample-chip");
  if (first) first.click();
}

async function loadSample(key, btn) {
  document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
  if (btn) btn.classList.add("active");
  const model = $("modelSel").value;
  const j = await api(`/api/samples/${key}/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  attachJob(j.job_id);
}

$("uploadBtn").onclick = async () => {
  const inp = $("fileInput");
  if (!inp.files.length) {
    alert("请先选择一个视频文件");
    return;
  }
  const fd = new FormData();
  fd.append("video", inp.files[0]);
  fd.append("model", $("modelSel").value);
  $("uploadBtn").disabled = true;
  $("uploadBtn").textContent = "上传中…";
  try {
    const j = await api("/api/upload", { method: "POST", body: fd });
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
    attachJob(j.job_id);
  } catch (e) {
    showFatal(`上传失败：${e.message}`);
  } finally {
    $("uploadBtn").disabled = false;
    $("uploadBtn").textContent = "上传并稳定";
  }
};

$("modelSel").onchange = async () => {
  if (!state.jobId || !state.job) return;
  try {
    if (state.job.stage === "analyzing" || state.job.stage === "rendering") return;
    await api(`/api/jobs/${state.jobId}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: $("modelSel").value }),
    });
    attachJob(state.jobId);
  } catch (e) {
    showFatal(`切换模型失败：${e.message}`);
  }
};

/* ------------------------------------------------------- job polling */

function attachJob(jid) {
  state.jobId = jid;
  state.loadedVersion = null;
  state.plan = null;
  state.lastDrawnFrame = -1;
  $("jobBanner").classList.remove("hidden");
  $("errorBox").classList.add("hidden");
  $("stabEmpty").classList.remove("hidden");
  $("srcEmpty").classList.remove("hidden");
  $("playBtn").disabled = true;
  $("seekBar").disabled = true;
  pollJob();
}

function showFatal(msg) {
  $("jobBanner").classList.remove("hidden");
  const eb = $("errorBox");
  eb.classList.remove("hidden");
  eb.textContent = msg;
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const j = await api(`/api/jobs/${state.jobId}`);
    state.job = j;
    renderJobBanner(j);

    if (j.stage === "done" && j.has_result) {
      $("playBtn").disabled = false;
      $("seekBar").disabled = false;
      if (state.loadedVersion !== j.video_version) {
        await loadResult(j);
      }
      if (state.renderQueued && j.stage === "done") {
        state.renderQueued = false;
        renderNow();
      }
    }
    if (j.stage === "error_analysis" || j.stage === "error_render") {
      if (j.error) {
        const eb = $("errorBox");
        eb.classList.remove("hidden");
        eb.innerHTML =
          `<b>${j.stage === "error_analysis" ? "分析失败" : "渲染失败"}：</b> ${escapeHtml(j.error)}` +
          (j.stage === "error_render"
            ? ` <button class="btn tiny" id="retryRender">用当前参数重试</button>`
            : "");
        const rb = $("retryRender");
        if (rb) rb.onclick = () => renderNow();
      }
      state.renderQueued = false;
    }
  } catch (e) {
    // transient polling errors are non-fatal
    $("stageText").textContent = `状态查询失败：${e.message}，重试中…`;
  }
}

function renderJobBanner(j) {
  const stageMap = {
    queued: "排队中",
    analyzing: "① 特征匹配与运动分析",
    ready: "分析完成，准备渲染",
    rendering: "② 轨迹平滑 · 视野裁决 · 重采样编码",
    done: "完成 · 可播放 / 微调参数",
    error_analysis: "运动分析失败",
    error_render: "渲染失败（可重试）",
  };
  $("stageText").textContent = `${stageMap[j.stage] || j.stage} — ${j.message || ""}`;
  const pct = Math.round((j.progress || 0) * 100);
  $("progressBar").style.width = pct + "%";
  $("progressPct").textContent = pct + "%";
  $("renderingPill").classList.toggle("hidden", j.stage !== "rendering");
}

setInterval(() => {
  if (state.jobId) pollJob();
}, 900);

async function loadResult(j) {
  const data = await api(`/api/jobs/${state.jobId}/result`);
  state.result = data;
  state.plan = { frames: data.frames, metrics: data.metrics, params: data.params };
  state.loadedVersion = j.video_version;

  if (j.source_url) {
    const src = $("srcVideo");
    if (!src.src.endsWith(j.source_url)) {
      src.src = j.source_url;
      src.load();
    }
  }
  const stab = $("stabVideo");
  const wantT = stab.currentTime || 0;
  const wasPlaying = !stab.paused;
  if (j.video_url) {
    stab.src = j.video_url;
    stab.load();
    stab.addEventListener(
      "loadedmetadata",
      () => {
        try {
          stab.currentTime = Math.min(wantT, stab.duration || 0);
          if (wasPlaying) stab.play().catch(() => {});
        } catch (_) {}
      },
      { once: true }
    );
    $("downloadBtn").classList.remove("disabled");
    $("downloadBtn").href = `/api/jobs/${state.jobId}/stabilized.mp4`;
  }
  $("smoothRange").value = data.params.smoothness;
  $("smoothVal").textContent = data.params.smoothness;
  $("fovRange").value = Math.round(data.params.min_fov * 100);
  $("fovVal").textContent = Math.round(data.params.min_fov * 100) + "%";
  updateZoomHint();
  renderAll();
}

/* ---------------------------------------------------- parameter edits */

let planDeb = null;
let renderDeb = null;

function updateZoomHint() {
  const fov = $("fovRange").value / 100;
  $("zoomCapHint").textContent = `最大放大 ≈ ${(1 / fov).toFixed(2)}×`;
  $("fovVal").textContent = $("fovRange").value + "%";
}

$("smoothRange").addEventListener("input", () => {
  $("smoothVal").textContent = $("smoothRange").value;
  schedulePlan();
});
$("fovRange").addEventListener("input", () => {
  updateZoomHint();
  schedulePlan();
});

function currentParams() {
  return {
    smoothness: Number($("smoothRange").value),
    min_fov: Number($("fovRange").value) / 100,
  };
}

function schedulePlan() {
  clearTimeout(planDeb);
  planDeb = setTimeout(postPlan, 180);
  if ($("autoRender").checked) {
    clearTimeout(renderDeb);
    renderDeb = setTimeout(() => renderNow(), 1100);
  }
}

async function postPlan() {
  const okStages = ["ready", "rendering", "done", "error_render"];
  if (!state.jobId || !state.job || !okStages.includes(state.job.stage)) {
    return;
  }
  try {
    const p = currentParams();
    const data = await api(`/api/jobs/${state.jobId}/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
    });
    state.plan = data;
    renderMetrics(data.metrics, null);
    drawChartsOnly();
    drawOverlay();
  } catch (e) {
    $("mHfSub").textContent = `规划失败：${e.message}`;
  }
}

$("renderBtn").onclick = () => renderNow(true);

async function renderNow(manual) {
  if (!state.jobId) return;
  if (state.job && ["analyzing", "rendering"].includes(state.job.stage)) {
    state.renderQueued = true;
    return;
  }
  clearTimeout(renderDeb);
  try {
    await api(`/api/jobs/${state.jobId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentParams()),
    });
    pollJob();
  } catch (e) {
    if (String(e.message).includes("已有渲染任务")) {
      state.renderQueued = true;
    } else if (manual) {
      showFatal(`渲染请求失败：${e.message}`);
    }
  }
}

/* ------------------------------------------------------ transport/sync */

const srcV = $("srcVideo");
const stabV = $("stabVideo");

srcV.addEventListener("loadeddata", () => $("srcEmpty").classList.add("hidden"));
stabV.addEventListener("loadeddata", () => $("stabEmpty").classList.add("hidden"));

$("playBtn").onclick = () => {
  if (stabV.paused) stabV.play().catch(() => {});
  else stabV.pause();
};
stabV.addEventListener("play", () => {
  srcV.play().catch(() => {});
  $("playBtn").textContent = "⏸ 暂停（双画面同步）";
});
stabV.addEventListener("pause", () => {
  srcV.pause();
  $("playBtn").textContent = "▶ 播放 / 暂停（双画面同步）";
});
stabV.addEventListener("seeked", () => {
  if (Math.abs(srcV.currentTime - stabV.currentTime) > 0.05) {
    srcV.currentTime = stabV.currentTime;
  }
});

$("seekBar").addEventListener("input", () => {
  if (!state.plan) return;
  const dur = stabV.duration || srcV.duration || 0;
  const t = ($("seekBar").value / 1000) * dur;
  stabV.currentTime = t;
  srcV.currentTime = t;
});

function seekToFrame(i) {
  if (!state.plan) return;
  const fps = state.job?.fps || 30;
  const t = i / fps;
  stabV.currentTime = t;
  srcV.currentTime = t;
}

function frameIndexNow() {
  const n = state.plan?.frames?.length || 0;
  if (!n) return 0;
  const fps = state.job?.fps || 30;
  let i = Math.round(srcV.currentTime * fps);
  if (!Number.isFinite(i)) i = 0;
  return Math.max(0, Math.min(n - 1, i));
}

function tick() {
  const n = state.plan?.frames?.length || 0;
  if (n) {
    // drift correction
    if (!stabV.paused && Math.abs(stabV.currentTime - srcV.currentTime) > 0.22) {
      stabV.currentTime = srcV.currentTime;
    }
    const f = frameIndexNow();
    state.frame = f;
    const dur = srcV.duration || stabV.duration || 0;
    if (dur > 0 && document.activeElement !== $("seekBar")) {
      $("seekBar").value = String(Math.round((srcV.currentTime / dur) * 1000));
    }
    $("timeLabel").textContent =
      `${srcV.currentTime.toFixed(2)}s / ${dur.toFixed(2)}s · 帧 ${f}/${n - 1}`;
    if (f !== state.lastDrawnFrame) {
      if (drawOverlay()) state.lastDrawnFrame = f;
      drawChartCursors();
      updateReadout(f);
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/* --------------------------------------------------------- crop overlay */

function resizeOverlayCanvas() {
  const c = $("overlay");
  if (srcV.videoWidth) {
    c.width = srcV.videoWidth;
    c.height = srcV.videoHeight;
  }
}
srcV.addEventListener("loadedmetadata", () => {
  resizeOverlayCanvas();
  state.lastDrawnFrame = -1;
  drawOverlay();
});

function drawOverlay() {
  const c = $("overlay");
  if (!srcV.videoWidth) return false;
  if (c.width !== srcV.videoWidth) resizeOverlayCanvas();
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const fr = state.plan?.frames?.[state.frame];
  if (!fr || !fr.quad) return false;
  const q = fr.quad;

  // dim outside the valid quad
  ctx.beginPath();
  ctx.rect(0, 0, c.width, c.height);
  ctx.moveTo(q[0][0], q[0][1]);
  for (let k = 3; k >= 0; k--) ctx.lineTo(q[k][0], q[k][1]);
  ctx.closePath();
  ctx.fillStyle = "rgba(0,0,0,0.46)";
  ctx.fill("evenodd");

  // quad outline
  ctx.beginPath();
  ctx.moveTo(q[0][0], q[0][1]);
  q.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
  ctx.closePath();
  ctx.lineWidth = Math.max(2, c.width / 420);
  ctx.strokeStyle = fr.sacrificed ? "#f85149" : "#ffd34d";
  ctx.shadowColor = "rgba(0,0,0,.8)";
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;

  const r = Math.max(3, c.width / 180);
  q.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx.fillStyle = "#ffd34d";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#1b1b1b";
    ctx.stroke();
  });

  ctx.font = `${Math.round(c.width / 46)}px sans-serif`;
  const label = fr.sacrificed ? "有效区域（本帧已牺牲部分稳定度）" : "有效区域";
  const lx = Math.max(8, q[0][0]);
  const ly = Math.max(22, q[0][1] - 8);
  ctx.fillStyle = fr.sacrificed ? "#ff8b84" : "#ffd34d";
  ctx.fillText(label, lx, ly);
  return true;
}

/* -------------------------------------------------------------- charts */

function prepCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320;
  const h = canvas.clientHeight || 180;
  const W = Math.round(w * dpr);
  const H = Math.round(h * dpr);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function bounds(series, hlines) {
  let lo = Infinity, hi = -Infinity;
  series.forEach((s) => {
    if (s.axis === "r") return;
    s.data.forEach((v) => {
      if (v === null || v === undefined || Number.isNaN(v)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
  });
  (hlines || []).forEach((h) => {
    if (h.axis === "r") return;
    lo = Math.min(lo, h.v);
    hi = Math.max(hi, h.v);
  });
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function drawChart(canvas, cfg) {
  const { ctx, w, h } = prepCanvas(canvas);
  const padL = 46, padR = cfg.rightRange ? 40 : 12, padT = 10, padB = 18;
  const iw = w - padL - padR, ih = h - padT - padB;
  const n = cfg.n || Math.max(...cfg.series.map((s) => s.data.length), 2);
  const X = (i) => padL + (n <= 1 ? 0 : (i / (n - 1)) * iw);
  const [lo, hi] = bounds(cfg.series, cfg.hlines);
  const Y = (v) => padT + ih - ((v - lo) / (hi - lo)) * ih;
  const [rlo, rhi] = cfg.rightRange || [0, 1];
  const YR = (v) => padT + ih - ((v - rlo) / (rhi - rlo)) * ih;

  ctx.strokeStyle = "#222b36";
  ctx.fillStyle = "#7d8a99";
  ctx.font = "10.5px ui-monospace, monospace";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = padT + (ih * g) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    const val = hi - ((hi - lo) * g) / 4;
    ctx.textAlign = "right";
    ctx.fillText(formatTick(val), padL - 6, y + 3);
  }
  if (cfg.rightRange) {
    ctx.textAlign = "left";
    for (let g = 0; g <= 2; g++) {
      const v = rhi - ((rhi - rlo) * g) / 2;
      const y = padT + (ih * g) / 2;
      ctx.fillStyle = "#bc8cff";
      ctx.fillText(v.toFixed(1), w - padR + 5, y + 3);
    }
  }

  (cfg.hlines || []).forEach((hl) => {
    const y = hl.axis === "r" ? YR(hl.v) : Y(hl.v);
    ctx.setLineDash(hl.dash || [5, 4]);
    ctx.strokeStyle = hl.color || "#888";
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  cfg.series.forEach((s) => {
    const yy = s.axis === "r" ? YR : Y;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.6;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    s.data.forEach((v, i) => {
      if (v === null || v === undefined || Number.isNaN(v)) { pen = false; return; }
      const x = X(i), y = yy(v);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  });

  if (cfg.marks) {
    ctx.fillStyle = cfg.markColor || "#f85149";
    cfg.marks.forEach((i) => {
      const s = cfg.markSeries || 0;
      const ser = cfg.series[s];
      const v = ser.data[i];
      if (v === null || v === undefined) return;
      ctx.beginPath();
      ctx.arc(X(i), (ser.axis === "r" ? YR(v) : Y(v)), 2.6, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  const ci = state.hover ?? state.frame;
  if (ci >= 0 && ci < n) {
    ctx.strokeStyle = "rgba(230,237,243,.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(ci), padT);
    ctx.lineTo(X(ci), padT + ih);
    ctx.stroke();
  }

  // legend
  let lx = padL + 6;
  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  cfg.series.slice(0, 4).forEach((s) => {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, 2, 14, 3);
    ctx.fillStyle = "#9aa7b4";
    const label = s.label || "";
    ctx.fillText(label, lx + 18, 7);
    lx += 24 + ctx.measureText(label).width + 14;
  });

  canvas._chart = { X, n, padL, padR, w };
}

function formatTick(v) {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function indexFromEvent(canvas, ev) {
  const meta = canvas._chart;
  if (!meta) return 0;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const frac = (x - meta.padL) / (meta.w - meta.padL - meta.padR);
  return Math.max(0, Math.min(meta.n - 1, Math.round(frac * (meta.n - 1))));
}

["chartTraj", "chartFov", "chartResid"].forEach((id) => {
  const c = $(id);
  c.addEventListener("click", (e) => seekToFrame(indexFromEvent(c, e)));
  c.addEventListener("mousemove", (e) => {
    state.hover = indexFromEvent(c, e);
    drawChartsOnly();
  });
  c.addEventListener("mouseleave", () => {
    state.hover = null;
    drawChartsOnly();
  });
});

window.addEventListener("resize", () => {
  drawChartsOnly();
  drawOverlay();
});

$("trajSeg").querySelectorAll("button").forEach((b) => {
  b.onclick = () => {
    $("trajSeg").querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.trajCh = b.dataset.ch;
    drawChartsOnly();
  };
});

function drawChartsOnly() {
  if (!state.plan) return;
  drawTrajChart();
  drawFovChart();
  drawResidChart();
}
function drawChartCursors() { drawChartsOnly(); }

function col(frames, key) { return frames.map((f) => (f[key] === undefined ? null : f[key])); }

function drawTrajChart() {
  const fr = state.plan.frames;
  const ch = state.trajCh;
  const map = { x: ["ox", "tx"], y: ["oy", "ty"], a: ["oa", "ta"], s: ["os", "ts"] }[ch];
  drawChart($("chartTraj"), {
    n: fr.length,
    series: [
      { label: "原始路径", data: col(fr, map[0]), color: "#8b98a7", width: 1.4 },
      { label: "平滑目标", data: col(fr, map[1]), color: "#4493f8", width: 2 },
    ],
  });
}

function drawFovChart() {
  const fr = state.plan.frames;
  const minFov = currentParams().min_fov * 100;
  const sacIdx = fr.map((f, i) => (f.sacrificed ? i : -1)).filter((i) => i >= 0);
  drawChart($("chartFov"), {
    n: fr.length,
    rightRange: [0, 1.05],
    hlines: [{ v: minFov, color: "#f85149", dash: [6, 4] }],
    series: [
      { label: "视野%", data: fr.map((f) => f.fov * 100), color: "#3fb950", width: 1.8 },
      { label: "α", data: col(fr, "alpha"), color: "#bc8cff", width: 1.2, axis: "r" },
    ],
    marks: sacIdx,
    markSeries: 0,
  });
}

function drawResidChart() {
  const fr = state.plan.frames;
  drawChart($("chartResid"), {
    n: fr.length,
    rightRange: [0, 1],
    series: [
      { label: "全部残差", data: col(fr, "residual_all"), color: "#e8873a", width: 1.5 },
      { label: "内点RMSE", data: col(fr, "rmse_in"), color: "#3fb950", width: 1.3 },
      { label: "坏点比例", data: col(fr, "bad_fraction"), color: "#f85149", width: 1.1, axis: "r" },
    ],
  });
}

function updateReadout(i) {
  const f = state.plan?.frames?.[i];
  if (!f) return;
  $("trajReadout").textContent =
    `帧 ${i}：原始 (${f.ox}, ${f.oy}px, ${f.oa}°, ×${f.os}) → 目标 ` +
    `(${f.tx}, ${f.ty}px, ${f.ta}°, ×${f.ts})；zoom ×${f.zoom}，FOV ${(f.fov * 100).toFixed(1)}%，α=${f.alpha}` +
    (f.residual_all !== undefined ? `；残差 ${f.residual_all}px，内点 ${f.n_inliers}/${f.n_verified}` : "");
}

/* ------------------------------------------------------- metrics etc. */

function renderAll() {
  renderMetrics(state.plan.metrics, state.result?.verification || null);
  renderParallax(state.plan.metrics.parallax);
  renderWarnings(state.plan.metrics);
  drawChartsOnly();
  drawOverlay();
}

function renderMetrics(m, ver) {
  if (!m) return;
  $("mHfOrig").textContent = m.orig_hf.hf_index_px.toFixed(2);
  if (ver && ver.output_hf) {
    const out = ver.output_hf.hf_index_px;
    const red = ver.hf_reduction_pct;
    $("mHfOut").textContent = out.toFixed(3);
    const cls = red >= 70 ? "good" : red >= 30 ? "meh" : "badtext";
    $("mHfSub").innerHTML =
      `端到端实测削减 <b class="${cls}">${red}%</b>（对输出视频重新分析）`;
  } else {
    $("mHfOut").textContent = m.target_hf.hf_index_px.toFixed(3);
    $("mHfSub").textContent = "规划目标值；渲染后给出端到端实测";
  }
  const f = m.fov;
  $("mFov").textContent = `${(f.mean_retained_fov * 100).toFixed(1)}%`;
  $("mFovSub").textContent =
    `最低 ${(f.min_retained_fov * 100).toFixed(1)}% · 峰值放大 ×${f.max_zoom.toFixed(2)}`;
  $("mSac").textContent = `${f.sacrificed_frames}`;
  $("mSacSub").textContent =
    `/ ${state.plan.frames.length} 帧（${(f.sacrificed_fraction * 100).toFixed(0)}%）· 均 α=${f.mean_correction_strength.toFixed(2)}`;
  const an = m.analysis || {};
  $("mRmse").textContent =
    an.mean_inlier_rmse_px != null ? `${an.mean_inlier_rmse_px.toFixed(2)} px` : "–";
  $("mInlier").textContent =
    an.mean_inlier_ratio != null
      ? `平均内点率 ${(an.mean_inlier_ratio * 100).toFixed(1)}% · 插值帧 ${an.pairs_imputed || 0}`
      : "内点率 –";
}

function renderParallax(p) {
  const box = $("parallaxBanner");
  if (!p) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden", "high", "medium", "low");
  box.classList.add(p.level);
  const label = { high: "视差/模型残差明显", medium: "中等残差", low: "符合平面假设" }[p.level];
  box.innerHTML =
    `<b class="level">${label}</b> ${escapeHtml(p.message)}<br>` +
    `<span style="font-size:12px;opacity:.85">支持模型：${escapeHtml(p.supported_model)} · ` +
    `平均坏点比例 ${(p.mean_bad_fraction * 100).toFixed(1)}% · 空间残差标准差 ${p.mean_spatial_std_px}px · ` +
    `平均全残差 ${p.mean_model_residual_px}px</span>`;
}

function renderWarnings(m) {
  const area = $("warningArea");
  area.innerHTML = "";
  const imp = m.analysis?.pairs_imputed || 0;
  if (imp > 0) {
    const d = document.createElement("div");
    d.className = "warn-box info";
    d.textContent = `有 ${imp} 个帧间对缺少可靠特征，已用邻近匀速运动插值并在数据中标记；如成片局部异常请换用纹理更丰富的片段。`;
    area.appendChild(d);
  }
  if (m.coverage_violations > 0) {
    const d = document.createElement("div");
    d.className = "warn-box bad";
    d.textContent = `检测到 ${m.coverage_violations} 帧几何越界（潜在黑边），请提高视野下限或降低平滑强度后重新渲染。`;
    area.appendChild(d);
  }
}

/* ------------------------------------------------------------- benchmark */

async function loadBenchmark() {
  try {
    const r = await api("/api/benchmark");
    if (r.ready) renderBenchmark(r.report);
    else $("benchStatus").textContent = "尚无报告：点击“重新运行评测”，或在服务器执行 python3 scripts/run_benchmark.py";
  } catch (_) {}
}

function renderBenchmark(rep) {
  const tb = $("benchTable").querySelector("tbody");
  tb.innerHTML = "";
  const setName = { off: "关闭", mid: "标准55", max: "最强100", tightfov: "视野95%" };
  Object.entries(rep.clips).forEach(([key, clip]) => {
    const ids = ["off", "mid", "max", "tightfov"].filter((id) => clip.settings[id]);
    ids.forEach((id, rowIdx) => {
      const s = clip.settings[id];
      const tr = document.createElement("tr");
      if (rowIdx === 0) {
        const tdClip = document.createElement("td");
        tdClip.rowSpan = ids.length;
        const p = clip.parallax;
        tdClip.innerHTML =
          `<b>${clip.title}</b>` +
          (p ? ` <span class="pill ${p.level}">${p.level === "high" ? "视差高" : p.level === "medium" ? "残差中" : "平面"}</span>` : "") +
          `<span class="clip-desc">${clip.description}</span>`;
        tr.appendChild(tdClip);
      }
      const tdSet = document.createElement("td");
      tdSet.textContent = setName[id] || id;
      tr.appendChild(tdSet);

      const tdHf = document.createElement("td");
      tdHf.textContent = `${s.orig_hf_index_px} → ${s.output_hf_index_px}`;
      tr.appendChild(tdHf);

      const tdRed = document.createElement("td");
      tdRed.textContent = s.hf_reduction_pct == null ? "–" : s.hf_reduction_pct + "%";
      tdRed.className = s.hf_reduction_pct >= 70 ? "good" : s.hf_reduction_pct >= 30 ? "meh" : "";
      tr.appendChild(tdRed);

      const tdFov = document.createElement("td");
      tdFov.textContent = (s.fov.min_retained_fov * 100).toFixed(1) + "%";
      tr.appendChild(tdFov);

      const tdSac = document.createElement("td");
      tdSac.textContent = s.fov.sacrificed_frames;
      if (s.fov.sacrificed_fraction > 0.3) tdSac.className = "meh";
      tr.appendChild(tdSac);

      const tdV = document.createElement("td");
      tdV.textContent = s.coverage_violations;
      tdV.className = s.coverage_violations === 0 ? "good" : "badtext";
      tr.appendChild(tdV);

      if (rowIdx === 0) {
        const tdGt = document.createElement("td");
        tdGt.rowSpan = ids.length;
        const g = clip.gt_rmse;
        tdGt.innerHTML = g
          ? `平移 ${g.translation_rmse_px}px<br>角 ${g.angle_rmse_deg}°<br>缩放 ${g.scale_rmse_pct}%`
          : "–";
        tr.appendChild(tdGt);
      }
      tb.appendChild(tr);
    });
  });
  const extras = [];
  if (rep.clips.handheld_scene?.moving_object?.trend_corr != null) {
    const m = rep.clips.handheld_scene.moving_object;
    extras.push(`自制街景中红色车辆在稳定后仍单向运动：斜率 ${m.slope_px_per_frame} px/帧，趋势相关 ${m.trend_corr}（时间顺序/真实运动保留）。`);
  }
  extras.push(`报告生成于 ${rep.generated_at}，ffmpeg: ${rep.ffmpeg}。`);
  $("benchExtra").textContent = extras.join(" ");
  $("benchStatus").textContent = "";
}

$("runBenchBtn").onclick = async () => {
  $("runBenchBtn").disabled = true;
  $("benchStatus").textContent = "评测在后台运行，约 1 分钟…";
  try {
    await api("/api/benchmark/run", { method: "POST" });
    const t = setInterval(async () => {
      try {
        const st = await api("/api/benchmark/status");
        if (!st.running) {
          clearInterval(t);
          $("runBenchBtn").disabled = false;
          $("benchStatus").textContent = "评测完成，刷新报告中…";
          await loadBenchmark();
        }
      } catch (_) {}
    }, 2500);
  } catch (e) {
    $("runBenchBtn").disabled = false;
    $("benchStatus").textContent = "启动失败：" + e.message;
  }
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* --------------------------------------------------------------- init */
loadSamples().catch((e) => showFatal(`无法载入样片列表：${e.message}`));
loadBenchmark();
