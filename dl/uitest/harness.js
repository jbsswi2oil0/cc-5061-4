const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const ROOT = "/workspace";
const html = fs.readFileSync(path.join(ROOT, "web/templates/index.html"), "utf8");
const appjs = fs.readFileSync(path.join(ROOT, "web/static/app.js"), "utf8");

const errors = [];
const virtualConsole = new (require("jsdom").VirtualConsole)();
virtualConsole.on("jsdomError", (e) => {
  const msg = e.detail?.stack || e.message || "";
  if (msg.includes("Not implemented") && msg.includes("HTMLMediaElement")) return;
  errors.push("jsdomError: " + msg);
});
virtualConsole.on("error", (...a) => errors.push("console.error: " + a.join(" ")));

const dom = new JSDOM(html, {
  url: "http://127.0.0.1:8000/",
  runScripts: "outside-only",
  pretendToBeVisual: true,
  virtualConsole,
});
const { window } = dom;

// bridge fetch to Node's global fetch with absolute URLs
window.fetch = (url, opts) => {
  const u = new URL(url, "http://127.0.0.1:8000/").toString();
  return fetch(u, opts);
};

// canvas 2d context no-op proxy
function makeCtx() {
  return new Proxy(
    {},
    {
      get(t, p) {
        if (p === "measureText") return () => ({ width: 12 });
        if (p === "createLinearGradient")
          return () => ({ addColorStop() {} });
        if (p in t) return t[p];
        return typeof p === "string" ? () => {} : undefined;
      },
      set(t, p, v) {
        t[p] = v;
        return true;
      },
    }
  );
}
window.HTMLCanvasElement.prototype.getContext = function () {
  return makeCtx();
};
Object.defineProperty(window.HTMLCanvasElement.prototype, "clientWidth", {
  configurable: true,
  get() {
    return 520;
  },
});
Object.defineProperty(window.HTMLCanvasElement.prototype, "clientHeight", {
  configurable: true,
  get() {
    return this.id === "chartTraj" ? 210 : 150;
  },
});
window.HTMLMediaElement.prototype.play = () => Promise.resolve();
window.HTMLMediaElement.prototype.pause = () => {};

window.addEventListener("error", (e) => errors.push("window.error: " + e.message));

async function waitFor(fn, timeout = 40000, step = 400) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    try {
      const v = fn();
      if (v) return v;
    } catch (_) {}
    await new Promise((r) => setTimeout(r, step));
  }
  throw new Error("waitFor timeout: " + fn.toString().slice(0, 80));
}

(async () => {
  window.eval(appjs);

  await waitFor(() => window.document.querySelectorAll(".sample-chip").length === 3);
  console.log("PASS sample chips rendered: 3");

  // auto-loaded planar job -> metrics populated
  await waitFor(() => {
    const t = window.document.getElementById("mHfOut").textContent;
    return t && t !== "–";
  }, 45000);
  console.log("PASS initial result metrics rendered:",
    window.document.getElementById("mHfOrig").textContent, "->",
    window.document.getElementById("mHfOut").textContent,
    "|", window.document.getElementById("mHfSub").textContent.trim());

  await waitFor(() => window.document.querySelectorAll("#benchTable tbody tr").length >= 8);
  console.log("PASS benchmark rows:",
    window.document.querySelectorAll("#benchTable tbody tr").length);

  const pb = window.document.getElementById("parallaxBanner");
  if (pb.classList.contains("hidden")) throw new Error("parallax banner missing");
  console.log("PASS parallax banner level:", pb.classList.contains("low") ? "low" : "?");

  // charts got backing dimensions
  const cw = window.document.getElementById("chartTraj").width;
  if (!cw || cw < 100) throw new Error("chart canvas not sized: " + cw);
  console.log("PASS chart canvas sized width=", cw);

  // switch trajectory channel
  window.document.querySelector('#trajSeg button[data-ch="a"]').click();
  await new Promise((r) => setTimeout(r, 100));
  window.document.querySelector('#trajSeg button[data-ch="y"]').click();
  console.log("PASS channel switch clicks without error");

  // drag smoothness -> plan preview
  const sr = window.document.getElementById("smoothRange");
  sr.value = "95";
  sr.dispatchEvent(new window.Event("input", { bubbles: true }));
  await waitFor(() =>
    window.document.getElementById("mHfSub").textContent.includes("规划目标"), 5000);
  console.log("PASS slider plan preview:",
    window.document.getElementById("mFov").textContent,
    window.document.getElementById("mSac").textContent, "sac");

  // auto-render is checked -> wait for re-render completion cycle
  await new Promise((r) => setTimeout(r, 2500));
  await waitFor(() => {
    const pillHidden = window.document.getElementById("renderingPill").classList.contains("hidden");
    const stage = window.document.getElementById("stageText").textContent;
    return pillHidden && stage.includes("完成");
  }, 40000);
  await new Promise((r) => setTimeout(r, 1500));
  if (!window.document.getElementById("errorBox").classList.contains("hidden")) {
    throw new Error("error box visible: " + window.document.getElementById("errorBox").textContent);
  }
  console.log("PASS auto re-render completed, no error box");
  console.log("HF sub after render:",
    window.document.getElementById("mHfSub").textContent.trim());

  // seek bar interaction
  const seek = window.document.getElementById("seekBar");
  seek.value = "500";
  seek.dispatchEvent(new window.Event("input", { bubbles: true }));
  console.log("PASS seek input dispatched");

  if (errors.length) {
    console.log("\nJS ERRORS (" + errors.length + "):");
    errors.slice(0, 12).forEach((e) => console.log(" -", e.split("\n")[0]));
    process.exit(1);
  }
  console.log("\nUI_LOGIC_OK (jsdom; real video decoding/pixels not exercised)");
  process.exit(0);
})().catch((e) => {
  console.error("HARNESS FAIL:", e.message);
  if (errors.length) errors.slice(0, 10).forEach((x) => console.error("  ", x.split("\n")[0]));
  process.exit(1);
});
